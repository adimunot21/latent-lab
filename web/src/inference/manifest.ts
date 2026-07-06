/**
 * manifest.json types + loader. The manifest is the contract between the
 * Python export pipeline and the browser: normalization stats, env config,
 * the fixed PCA basis, the checkpoint registry, and per-file sha256 used to
 * integrity-check every downloaded artifact.
 */

import { hubUrl } from '../config'

export interface FileEntry {
  path: string
  bytes: number
  sha256: string
}

export interface CheckpointEntry {
  id: string
  label: string
  description: string
  files: {
    encoder: FileEntry
    predictor: FileEntry
    encoder_int8: FileEntry
    predictor_int8: FileEntry
  }
}

export interface Manifest {
  manifest_version: number
  latent_dim: number
  frame_size: number
  action_dim: number
  norm_stats: { frame_mean: number; frame_std: number }
  env_config: Record<string, number>
  pca: { components: number[][]; mean: number[]; explained_variance_ratio: number[] }
  lookup: { n_states: number; states: FileEntry; latents: FileEntry; dtype: string }
  checkpoints: CheckpointEntry[]
  quantization_note: string
  hf: { repo_id: string; revision: string | null }
}

export async function loadManifest(): Promise<Manifest> {
  const response = await fetch(hubUrl('manifest.json'))
  if (!response.ok) {
    throw new Error(`manifest fetch failed: ${response.status} ${response.statusText}`)
  }
  return (await response.json()) as Manifest
}

/** Download a bundle file and verify its sha256 against the manifest entry. */
export async function fetchVerified(entry: FileEntry): Promise<ArrayBuffer> {
  const response = await fetch(hubUrl(entry.path))
  if (!response.ok) {
    throw new Error(`fetch ${entry.path} failed: ${response.status}`)
  }
  const buffer = await response.arrayBuffer()
  const digest = await crypto.subtle.digest('SHA-256', buffer)
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('')
  if (hex !== entry.sha256) {
    throw new Error(`integrity check failed for ${entry.path}: sha256 ${hex} != ${entry.sha256}`)
  }
  return buffer
}

/**
 * Normalize a rendered uint8 frame into encoder input layout (1, 1, N, N):
 * (raw / 255 - frame_mean) / frame_std — THE formula shared with training.
 */
export function normalizeFrame(frame: Uint8Array, manifest: Manifest): Float32Array {
  const { frame_mean: mean, frame_std: std } = manifest.norm_stats
  const out = new Float32Array(frame.length)
  for (let i = 0; i < frame.length; i++) {
    out[i] = (frame[i] / 255 - mean) / std
  }
  return out
}
