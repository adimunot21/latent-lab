/**
 * Model source configuration. The browser fetches all model artifacts from a
 * PINNED Hugging Face revision (a commit hash, never a moving branch) — this
 * is the single place the pin lives.
 */

export const HF_REPO_ID = 'adimunot/latent-lab'

/** Commit hash of the published bundle (see PROJECTPLAN Phase 4). */
export const HF_REVISION = 'dcb3b25d238556f0ef20ae3b6e128211ff519cc6'

/** Resolve a bundle-relative path to a pinned download URL. */
export function hubUrl(path: string): string {
  return `https://huggingface.co/${HF_REPO_ID}/resolve/${HF_REVISION}/${path}`
}
