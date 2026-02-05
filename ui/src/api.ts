import { TreeSnapshot } from "./types";

export async function fetchTreeSnapshot(): Promise<TreeSnapshot> {
  const response = await fetch("/api/tree");
  if (!response.ok) {
    throw new Error(`Failed to fetch tree snapshot: ${response.status}`);
  }

  return response.json() as Promise<TreeSnapshot>;
}

export async function resetTree(): Promise<void> {
  const response = await fetch("/api/reset", {
    method: "POST"
  });

  if (!response.ok) {
    throw new Error(`Failed to reset tree: ${response.status}`);
  }
}
