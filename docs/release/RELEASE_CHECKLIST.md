# Release and evidence checklist

This checklist supports the repository's current research prototype. It does not certify a medical device or authorize clinical use.

## Evidence-package gate

- [ ] GitHub Actions fast validation and release validation pass on the evidence branch.
- [ ] Offline and profile-specific regression tests pass without changing test selection.
- [ ] Any browser capture comes from the actual application and uses synthetic or rights-cleared data.
- [ ] Any local filesystem-backend workaround is clearly distinguished from the standard S3/MinIO Compose stack.
- [ ] Paper metrics, Phase 4 proxy metrics, and Stage 5 optimized-profile systems evidence remain separately attributed.
- [ ] Any public use of dataset-derived results is cleared against the dataset terms.
- [ ] The repository still has no newly assigned software license unless the owner explicitly chooses one.
- [ ] No secrets, patient data, restricted manuscript files, dataset media, or local environment files enter the evidence package.

## Current limitations

- The accepted MIUCC 2026 manuscript is not present in the source repository, so paper-specific methods and metrics remain unverified here.
- No clinical validation, paired patient-level multimodal evaluation, prospective study, or native smartphone performance result is available.
- Current end-to-end container reproduction is **PENDING** while the pinned MinIO Docker Hub images are unavailable.
- Browser screenshots and a captioned walkthrough are **PENDING** until capture and playback checks finish.
