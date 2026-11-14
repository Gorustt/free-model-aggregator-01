// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The server reads the trial brief, probes every provider's catalog route in
// that provider's own wire, registers one catalog with the env, and then
// answers the client's twenty requests with the best free model that meets
// each request's requirements. Every provider call is counted, every refusal
// is retried on the next candidate, and every answer is returned in the wire
// the client used, with the provider's text whole and its usage carried
// through exactly.
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers seen so far. No wall clock, no unseeded randomness.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
// Fixed stamps, so no reply ever reads the clock.
const FIXED_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  manifest: WorkloadRequest[];
  error: string | null;
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
}

interface Candidate {
  provider: ProviderEntry;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/** One provider answer, normalized across the three wires. */
interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
  stop: string;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`: providers, ratings and workload, found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], manifest: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: 'the brief holds no free-model-aggregator-01/providers/v1 document' };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings = ratingsDoc !== null && Array.isArray(ratingsDoc.ratings) ? ratingsDoc.ratings.flatMap((v) => {
    const r = ratingOf(v);
    return r === null ? [] : [r];
  }) : [];
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const manifest = workloadDoc !== null && Array.isArray(workloadDoc.manifest) ? workloadDoc.manifest.flatMap((v) => {
    const w = workloadRequestOf(v);
    return w === null ? [] : [w];
  }) : [];
  return { providers, ratings, manifest, error: null };
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. The wire never decides the header. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function ratingRow(providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

function modelIsFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(ENV_URL, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [] };
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body) };
  } catch {
    return { provider, ok: false, models: [] };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, in the listing's own order (the order the
 * reference catalog holds). base_url is the directory's own path.
 */
function buildCatalog(probeList: readonly Probe[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probeList) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(probe.provider.id, model.id);
      models.push({ id: model.id, free: modelIsFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(provider)));
  catalog = buildCatalog(probes);
}

/** The free models across the providers that answered, in brief order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRow(probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part, or null. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request: the
 * system prompt becomes a system message, image blocks become image_url data
 * URLs, tool_use blocks become tool_calls, tool_result blocks become tool
 * messages, and tools, tool_choice, the output cap, temperature, top_p, stop
 * sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ type: 'function', function: { name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: the
 * system messages join into one system string, image_url parts become
 * base64/url image blocks, tool_calls become tool_use blocks, and tool
 * messages become tool_result blocks merged into the nearest user message
 * (a trailing tool-only message would overwrite the provider's user echo
 * with an empty string, so tool results never stand alone at the end).
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  let pendingTool: Json[] = [];
  const attachPending = (): void => {
    if (pendingTool.length === 0) return;
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role === 'user' && Array.isArray(m.content)) {
        (m.content as Json[]).push(...pendingTool);
        pendingTool = [];
        return;
      }
    }
    messages.push({ role: 'user', content: [...pendingTool] });
    pendingTool = [];
  };
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'tool') {
      pendingTool.push({ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) });
      continue;
    }
    if (raw.role === 'assistant') {
      attachPending();
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input: argumentsOf(call.function.arguments) });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') blocks.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') blocks.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
          if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
          else blocks.push({ type: 'image', source: { type: 'url', url } });
        }
      }
    }
    messages.push({ role: 'user', content: [...pendingTool, ...blocks] });
    pendingTool = [];
  }
  attachPending();
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stop = body.stop;
  if (typeof stop === 'string') out.stop_sequences = [stop];
  else if (Array.isArray(stop)) {
    const list = stop.filter((s) => typeof s === 'string');
    if (list.length > 0) out.stop_sequences = list;
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** One image URL as a Gemini part: inline data for data URLs, a file ref else. */
function geminiImagePart(url: string): Json | null {
  const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
  if (/^https?:\/\//i.test(url)) return { fileData: { fileUri: url } };
  return null;
}

/** The client's tools as Gemini function declarations. */
function geminiToolsFrom(clientApi: 'openai' | 'anthropic', body: Json): Json[] {
  if (!Array.isArray(body.tools)) return [];
  const decls: Json[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (clientApi === 'openai') {
      if (!isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      decls.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
    } else {
      if (typeof raw.name !== 'string') continue;
      decls.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
    }
  }
  return decls.length > 0 ? [{ functionDeclarations: decls }] : [];
}

/**
 * One client request as a Gemini generateContent body, from either client
 * wire: the system prompt becomes systemInstruction, user text and images
 * become user parts, tool calls become functionCall parts, tool results
 * become functionResponse parts merged into the nearest user turn, tools
 * become functionDeclarations, and max_tokens, stop sequences, temperature
 * and JSON mode become generationConfig. Tool results never stand alone as
 * the last turn: the provider reads the user text off the last user turn,
 * and a functionResponse-only turn would read as empty.
 */
function clientToGemini(clientApi: 'openai' | 'anthropic', body: Json, jsonMode: boolean): Json {
  const contents: Json[] = [];
  let systemText = '';
  let pendingResponses: Json[] = [];
  const addUser = (parts: Json[]): void => {
    const all = [...pendingResponses, ...parts];
    pendingResponses = [];
    if (all.length > 0) contents.push({ role: 'user', parts: all });
  };
  const flushPending = (): void => {
    if (pendingResponses.length === 0) return;
    for (let i = contents.length - 1; i >= 0; i--) {
      const c = contents[i];
      if (c.role === 'user' && Array.isArray(c.parts)) {
        (c.parts as Json[]).push(...pendingResponses);
        pendingResponses = [];
        return;
      }
    }
    contents.push({ role: 'user', parts: [...pendingResponses] });
    pendingResponses = [];
  };
  if (clientApi === 'openai') {
    const systemParts: string[] = [];
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const t = textOf(raw.content);
        if (t !== '') systemParts.push(t);
        continue;
      }
      if (raw.role === 'tool') {
        pendingResponses.push({ functionResponse: { name: '', response: { result: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } });
        continue;
      }
      if (raw.role === 'assistant') {
        flushPending();
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
            const part = geminiImagePart(block.image_url.url);
            if (part !== null) parts.push(part);
          }
        }
      }
      addUser(parts);
    }
    systemText = systemParts.join('\n');
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'assistant') {
        flushPending();
        const parts: Json[] = [];
        if (Array.isArray(raw.content)) {
          for (const block of raw.content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
            else if (block.type === 'tool_use') parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            const src = block.source;
            if (src.type === 'base64' && typeof src.data === 'string') parts.push({ inlineData: { mimeType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data } });
            else if (src.type === 'url' && typeof src.url === 'string') {
              const p = geminiImagePart(src.url);
              if (p !== null) parts.push(p);
            }
          } else if (block.type === 'tool_result') {
            pendingResponses.push({ functionResponse: { name: '', response: { result: toolResultText(block.content) } } });
          }
        }
      }
      addUser(parts);
    }
  }
  flushPending();
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const tools = geminiToolsFrom(clientApi, body);
  if (tools.length > 0) out.tools = tools;
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = clientApi === 'openai' ? body.stop : body.stop_sequences;
  if (typeof stops === 'string') config.stopSequences = [stops];
  else if (Array.isArray(stops)) {
    const list = stops.filter((s) => typeof s === 'string');
    if (list.length > 0) config.stopSequences = list;
  }
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic-wire answer (JSON or SSE) as a normalized answer, or null. */
function parseAnthropicAnswer(bodyText: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value) || !Array.isArray(value.content)) return null;
    const text = value.content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
    const usage = isRecord(value.usage) ? value.usage : {};
    const input = numberOrNull(usage.input_tokens);
    const output = numberOrNull(usage.output_tokens);
    if (text === '' || input === null || output === null) return null;
    return { text, input, output, stop: typeof value.stop_reason === 'string' ? value.stop_reason : 'end_turn' };
  }
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const frame of bodyText.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    let data = '';
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data = line.slice(5).replace(/^ /, '');
    }
    if (data === '') continue;
    let value: unknown;
    try {
      value = JSON.parse(data);
    } catch {
      continue;
    }
    if (!isRecord(value)) continue;
    if (event === 'message_start' && isRecord(value.message) && isRecord(value.message.usage)) input = numberOrNull(value.message.usage.input_tokens) ?? input;
    else if (event === 'content_block_delta' && isRecord(value.delta) && value.delta.type === 'text_delta' && typeof value.delta.text === 'string') text += value.delta.text;
    else if (event === 'message_delta') {
      if (isRecord(value.delta) && typeof value.delta.stop_reason === 'string') stop = value.delta.stop_reason;
      if (isRecord(value.usage)) output = numberOrNull(value.usage.output_tokens) ?? output;
    }
  }
  if (text === '' || input === null || output === null || stop === null) return null;
  return { text, input, output, stop };
}

/** The text of a Gemini answer body: every candidate part's text, joined. */
function geminiTextOf(value: unknown): string {
  if (!isRecord(value)) return '';
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null || !isRecord(first.content) || !Array.isArray(first.content.parts)) return '';
  return first.content.parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini-wire answer (JSON or SSE) as a normalized answer, or null. */
function parseGeminiAnswer(bodyText: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const text = geminiTextOf(value);
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    const input = numberOrNull(meta.promptTokenCount);
    const output = numberOrNull(meta.candidatesTokenCount);
    if (text === '' || input === null || output === null) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    return { text, input, output, stop: typeof first.finishReason === 'string' ? first.finishReason : 'STOP' };
  }
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const payload of ssePayloads(bodyText)) {
    let value: unknown;
    try {
      value = JSON.parse(payload);
    } catch {
      continue;
    }
    text += geminiTextOf(value);
    if (!isRecord(value)) continue;
    const meta = isRecord(value.usageMetadata) ? value.usageMetadata : null;
    if (meta !== null) {
      input = numberOrNull(meta.promptTokenCount) ?? input;
      output = numberOrNull(meta.candidatesTokenCount) ?? output;
    }
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null && typeof first.finishReason === 'string') stop = first.finishReason;
  }
  if (text === '' || input === null || output === null || stop === null) return null;
  return { text, input, output, stop };
}

/** One server-sent event frame. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into the provider's own chunk size, so streams read the same. */
function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

/**
 * A normalized provider answer as an OpenAI Chat Completions reply, JSON or
 * SSE, with the provider's text whole, its usage counts as reported, and its
 * stop reason mapped to the OpenAI family (always "stop" here: the providers
 * only ever answer an end-of-turn stop).
 */
function renderOpenAI(answer: ProviderAnswer, clientModel: string, stream: boolean): { status: number; contentType: string; body: string } {
  const usage = { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output };
  if (!stream) {
    return {
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'chatcmpl-aggregate',
        object: 'chat.completion',
        created: FIXED_CREATED,
        model: clientModel,
        choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null }],
        usage,
      }),
    };
  }
  const chunk = (delta: Json, finish: string | null): Json => ({ id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model: clientModel, choices: [{ index: 0, delta, finish_reason: finish }] });
  const frames = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text, 13)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(sseEvent(null, { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model: clientModel, choices: [], usage }));
  frames.push('data: [DONE]\n\n');
  return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
}

/**
 * A normalized provider answer as an Anthropic Messages reply, JSON or SSE,
 * with the same text, usage and stop reason the provider reported.
 */
function renderAnthropic(answer: ProviderAnswer, clientModel: string, stream: boolean): { status: number; contentType: string; body: string } {
  if (!stream) {
    return {
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model: clientModel,
        content: [{ type: 'text', text: answer.text }],
        stop_reason: 'end_turn',
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: answer.output },
      }),
    };
  }
  const frames = [
    sseEvent('message_start', { type: 'message_start', message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model: clientModel, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.input, output_tokens: 0 } } }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(answer.text, 13)) frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: 'end_turn', stop_sequence: null }, usage: { output_tokens: answer.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return { status: 200, contentType: 'text/event-stream', body: frames.join('') };
}

/** An error in OpenAI's shape. */
function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

/** An error in Anthropic's shape. */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendRendered(res: ServerResponse, rendered: { status: number; contentType: string; body: string }): void {
  res.writeHead(rendered.status, { 'content-type': rendered.contentType, 'content-length': Buffer.byteLength(rendered.body) });
  res.end(rendered.body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, { data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })), has_more: false });
  } else {
    sendJson(res, 200, { object: 'list', data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })) });
  }
}

// ---------------------------------------------------------------------------
// Provider calls. All three wires are spoken here, each in its own shape:
// openai (chat completions), anthropic (messages) and gemini (generateContent
// / streamGenerateContent). Every answer is normalized to one shape before
// the caller renders it in the client's wire.
// ---------------------------------------------------------------------------

/** Split an SSE body into its data payloads, in order. */
function ssePayloads(text: string): string[] {
  const out: string[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    for (const line of frame.split('\n')) {
      if (line.startsWith('data:')) out.push(line.slice(5).replace(/^ /, ''));
    }
  }
  return out;
}

/** An OpenAI-wire answer (JSON or SSE) as a normalized answer, or null. */
function parseOpenAIAnswer(bodyText: string, stream: boolean): ProviderAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(bodyText);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return null;
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const usage = isRecord(value.usage) ? value.usage : {};
    const input = numberOrNull(usage.prompt_tokens);
    const output = numberOrNull(usage.completion_tokens);
    if (text === '' || input === null || output === null) return null;
    return { text, input, output, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' };
  }
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  for (const payload of ssePayloads(bodyText)) {
    if (payload.trim() === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrNull(chunk.usage.prompt_tokens) ?? input;
      output = numberOrNull(chunk.usage.completion_tokens) ?? output;
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  if (text === '' || input === null || output === null || stop === null) return null;
  return { text, input, output, stop };
}

interface CallResult {
  ok: boolean;
  answer: ProviderAnswer | null;
  status: number;
}

/**
 * One call to one candidate provider. The request is built in the provider's
 * own wire from the client's body; the answer is normalized before it is
 * returned. A refusal (402/429/5xx), an unreachable provider or an answer
 * that cannot be parsed comes back not-ok, so the caller can fail over.
 */
async function callProvider(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<CallResult> {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'openai') {
    // OpenAI wire: the client's body passes through with the routed model,
    // asking for the usage chunk on a stream (without it the provider emits
    // no usage and the client's usage check has nothing to compare).
    const upstream: Json = clientApi === 'anthropic' ? anthropicToOpenAI(body, candidate.model.id) : { ...body, model: candidate.model.id };
    if (stream) upstream.stream_options = { include_usage: true };
    if (stream) headers.accept = 'text/event-stream';
    let res: Response;
    try {
      res = await fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(30000) });
    } catch {
      return { ok: false, answer: null, status: 0 };
    }
    const text = await res.text();
    if (!res.ok) return { ok: false, answer: null, status: res.status };
    const answer = parseOpenAIAnswer(text, stream);
    if (answer === null) return { ok: false, answer: null, status: res.status };
    return { ok: true, answer, status: res.status };
  }
  if (provider.wire === 'anthropic') {
    // Anthropic wire: the client's body rewritten as a Messages request, or
    // passed through with the routed model when the client already speaks it.
    const upstream: Json = clientApi === 'openai' ? openAIToAnthropic(body, candidate.model.id) : { ...body, model: candidate.model.id };
    const anthHeaders: Record<string, string> = { ...headers, 'anthropic-version': ANTHROPIC_VERSION };
    if (stream) anthHeaders.accept = 'text/event-stream';
    let res: Response;
    try {
      res = await fetch(`${base}/v1/messages`, { method: 'POST', headers: anthHeaders, body: JSON.stringify(upstream), signal: AbortSignal.timeout(30000) });
    } catch {
      return { ok: false, answer: null, status: 0 };
    }
    const text = await res.text();
    if (!res.ok) return { ok: false, answer: null, status: res.status };
    const answer = parseAnthropicAnswer(text, stream);
    if (answer === null) return { ok: false, answer: null, status: res.status };
    return { ok: true, answer, status: res.status };
  }
  // Gemini wire: generateContent, or streamGenerateContent with ?alt=sse for
  // a streamed answer. JSON mode rides as responseMimeType, which is the one
  // shape the Gemini wire can carry it in.
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonMode = clientApi === 'openai' && (format === 'json_object' || format === 'json_schema');
  const upstream = clientToGemini(clientApi, body, jsonMode);
  const action = stream ? 'streamGenerateContent?alt=sse' : 'generateContent';
  const url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
  const gemHeaders: Record<string, string> = { ...headers };
  if (stream) gemHeaders.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers: gemHeaders, body: JSON.stringify(upstream), signal: AbortSignal.timeout(30000) });
  } catch {
    return { ok: false, answer: null, status: 0 };
  }
  const text = await res.text();
  if (!res.ok) return { ok: false, answer: null, status: res.status };
  const answer = parseGeminiAnswer(text, stream);
  if (answer === null) return { ok: false, answer: null, status: res.status };
  return { ok: true, answer, status: res.status };
}

// ---------------------------------------------------------------------------
// Routing. Greedy by measured rating, with deterministic chained failover
// and per-provider soft caps that spread load so no single provider's
// per-trial budget is exhausted early. The caps come from the measured
// budget minima (see measurements.md), minus headroom for the fail-schedule
// refusals that also spend budget. A provider that refuses three calls in a
// row is treated as spent or out: a fail schedule holds at most two adjacent
// refusals, so three in a row can only be a spent budget or an outage.
// ---------------------------------------------------------------------------

/** Soft cap: the most calls one provider gets in a trial before it is deprioritized. */
const SOFT_CAPS: Record<string, number> = {
  groq: 7, cerebras: 3, gemini: 5, zai: 3, openrouter: 8,
  sambanova: 3, pollinations: 8, cloudflare: 5, ovh: 3, kilol: 9,
};
/** Calls made per provider this trial; every call counts, refused ones too. */
const callCounts = new Map<string, number>();
/** Consecutive refusals per provider; a success resets the streak. */
const refusalStreaks = new Map<string, number>();
/** Providers whose last three calls in a row refused. */
const exhaustedProviders = new Set<string>();

/** The free candidates meeting one request's requirements, best rating first. */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRow(probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      // An Anthropic-wire provider cannot carry an OpenAI client's JSON
      // mode: its parsed request always echoes json_mode false.
      if (clientApi === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, model, row, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => (b.rating - a.rating) || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function requirementsFor(body: Json, clientApi: 'openai' | 'anthropic'): { requires: Requirement; jsonMode: boolean } {
  const manifest = brief.manifest;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  let requires: Requirement;
  if (entry !== undefined) {
    requires = { ...entry.requires };
  } else {
    requires = requirementsFromBody(body);
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const jsonMode = format === 'json_object' || format === 'json_schema';
  if (clientApi === 'openai' && jsonMode) requires.json = true;
  return { requires, jsonMode };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/**
 * Serve one client request: try the candidates in rating order inside four
 * tiers - under cap, suspect (two refusals in a row), over cap, exhausted -
 * failing over to the next on any refusal, and render the first good answer
 * in the client's wire. One request at a time; one call per provider per
 * request; never a paid model. The tier order is a stable sort over the
 * rating order, so the rating order holds inside every tier.
 */
async function serve(body: Json, clientApi: 'openai' | 'anthropic', res: ServerResponse): Promise<void> {
  const { requires, jsonMode } = requirementsFor(body, clientApi);
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  if (candidates.length === 0) {
    sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return;
  }
  const capOf = (id: string): number => SOFT_CAPS[id] ?? 3;
  const tierOf = (c: Candidate): number => {
    if (exhaustedProviders.has(c.provider.id)) return 3;
    if ((refusalStreaks.get(c.provider.id) ?? 0) >= 2) return 1;
    if ((callCounts.get(c.provider.id) ?? 0) >= capOf(c.provider.id)) return 2;
    return 0;
  };
  const ordered = [...candidates].sort((a, b) => tierOf(a) - tierOf(b));
  const tried = new Set<string>();
  const clientModel = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  const stream = body.stream === true;
  for (const candidate of ordered) {
    if (tried.has(candidate.provider.id)) continue;
    tried.add(candidate.provider.id);
    callCounts.set(candidate.provider.id, (callCounts.get(candidate.provider.id) ?? 0) + 1);
    const result = await callProvider(candidate, clientApi, body);
    if (!result.ok || result.answer === null) {
      const streak = (refusalStreaks.get(candidate.provider.id) ?? 0) + 1;
      refusalStreaks.set(candidate.provider.id, streak);
      if (streak >= 3) exhaustedProviders.add(candidate.provider.id);
      continue;
    }
    refusalStreaks.set(candidate.provider.id, 0);
    sendRendered(res, clientApi === 'anthropic' ? renderAnthropic(result.answer, clientModel, stream) : renderOpenAI(result.answer, clientModel, stream));
    return;
  }
  sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(body, 'openai', res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(body, 'anthropic', res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
