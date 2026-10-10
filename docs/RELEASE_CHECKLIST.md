# README and release checklist

The repository starts with documentation and paper figures. Use this checklist when uploading code and updating the [main README](../README.md).

## Paper and project metadata

- [ ] Confirm the final title, author order, affiliations, and contribution markers.
- [ ] Replace the paper status badge with the verified public paper URL.
- [ ] Add a project-page link if one is published.
- [ ] Replace the preliminary repository citation with the official paper BibTeX.
- [ ] Refresh the overview and architecture figures when the manuscript changes.

## Models and data

- [ ] Add working model links and model cards for each released checkpoint.
- [ ] Document the supported tasks, backbone, checkpoint format, and license for each model.
- [ ] Document training-data availability and preparation without implying that unreleased data can be downloaded.
- [ ] Keep model weights, datasets, video files, and credentials out of Git commits.

## Code and usage

- [ ] Upload the intended inference, evaluation, and training code.
- [ ] Replace the README's planned directory names with the actual code layout.
- [ ] Add dependency files and a tested installation sequence.
- [ ] Add minimal working grounding and QA examples with expected input and output formats.
- [ ] Document checkpoint paths, device requirements, and Cached / Joint inference options.
- [ ] Add benchmark download, preparation, prediction, and scoring commands.
- [ ] Document two-stage training configurations and launch commands if training code is released.
- [ ] Add the selected license and any required third-party notices.

## Results and final verification

- [ ] Keep README metrics and figures consistent with the final paper's Tables 1 and 2.
- [ ] Identify TimeLens-Bench refined annotations for grounding and M-Avg for MLVU.
- [ ] Preserve the latency scope: H200, batch size 1, model forward time, shared video computation, and the full Charades / LVBench evaluation sets.
- [ ] Check all links and images in GitHub's rendered README.
- [ ] Run every documented command against the released code and checkpoint versions.
- [ ] Update release checkboxes, status badges, and dated news entries only when the corresponding resources are available.
- [ ] Resolve applicable `TODO(...)` comments in the README.
