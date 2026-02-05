import fs from "node:fs";
import path from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";

const CWD = process.cwd();
const DEFAULT_TOKENIZER_PATHS = [path.resolve(CWD, "tokenizer", "tokenizer.json"), path.resolve(CWD, "tokenizer.json")];
const DEFAULT_TOKENIZER_CONFIG_PATHS = [
  path.resolve(CWD, "tokenizer", "tokenizer_config.json"),
  path.resolve(CWD, "tokenizer_config.json")
];

function readJson(filePath: string): Record<string, unknown> {
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as Record<string, unknown>;
}

function candidatePaths(fromEnv: string | undefined, defaults: string[]): string[] {
  const resolvedEnv = fromEnv
    ? [path.isAbsolute(fromEnv) ? fromEnv : path.resolve(CWD, fromEnv)]
    : [];
  return [...new Set([...resolvedEnv, ...defaults])];
}

function resolveExistingPath(label: string, fromEnv: string | undefined, defaults: string[]): string {
  const candidates = candidatePaths(fromEnv, defaults);
  const existing = candidates.find((candidate) => fs.existsSync(candidate));

  if (existing) {
    return existing;
  }

  throw new Error(
    [
      `Llama 3 ${label} file not found.`,
      `Checked: ${candidates.join(", ")}`,
      `Working directory: ${CWD}`
    ].join(" ")
  );
}

export async function loadLlamaTokenizer(): Promise<Tokenizer> {
  const tokenizerPath = resolveExistingPath("tokenizer", process.env.LLAMA3_TOKENIZER_PATH, DEFAULT_TOKENIZER_PATHS);
  const tokenizerConfigPath = resolveExistingPath(
    "tokenizer_config",
    process.env.LLAMA3_TOKENIZER_CONFIG_PATH,
    DEFAULT_TOKENIZER_CONFIG_PATHS
  );

  return new Tokenizer(readJson(tokenizerPath), readJson(tokenizerConfigPath));
}

export function decodeTokens(tokenizer: Tokenizer, tokens: number[]): string {
  return tokenizer.decode(tokens, { skip_special_tokens: true });
}
