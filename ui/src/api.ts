import { TreeSnapshot } from "./types";

function getApiBase(): string {
  const value = import.meta.env.VITE_MCTS_API_BASE as string | undefined;
  return (value ?? "").trim().replace(/\/$/, "");
}

function apiUrl(path: string): string {
  const base = getApiBase();
  return base ? `${base}${path}` : path;
}

export async function fetchTreeSnapshot(): Promise<TreeSnapshot> {
  const response = await fetch(apiUrl("/api/tree"));
  if (!response.ok) {
    throw new Error(`Failed to fetch tree snapshot: ${response.status}`);
  }

  return response.json() as Promise<TreeSnapshot>;
}

export async function resetTree(): Promise<void> {
  const response = await fetch(apiUrl("/api/reset"), {
    method: "POST"
  });

  if (!response.ok) {
    throw new Error(`Failed to reset tree: ${response.status}`);
  }
}

export async function resetWorkerTree(workerId: string): Promise<void> {
  const response = await fetch(apiUrl(`/api/reset-tree/${encodeURIComponent(workerId)}`), {
    method: "POST"
  });

  if (!response.ok) {
    throw new Error(`Failed to reset worker tree '${workerId}': ${response.status}`);
  }
}
