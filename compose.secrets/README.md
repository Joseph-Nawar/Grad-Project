# Local Compose secrets

`python scripts/stack.py up` creates the ignored `.runtime-secrets/` directory and populates it with random local-demo values. Do not create or commit secret files in this directory. The Compose file mounts them as Docker secrets; they are not placed in image layers or normal environment output.
