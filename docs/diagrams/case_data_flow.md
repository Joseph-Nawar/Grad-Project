# Case data flow

```mermaid
flowchart TD
  Collector[Collector UI] --> AssessAPI[Assessment API]
  AssessAPI --> Service[AssessmentService and four modality adapters]
  Service --> Fusion[Late fusion and deterministic symptom safeguard]
  Fusion --> CentralSnapshot[Immutable assessment snapshot]
  CentralSnapshot --> Central[(PostgreSQL case and assessment records)]
  CentralSnapshot --> Attachments[(MinIO attachments)]
  Edge[Edge collector] --> Offline[Offline workflow and local assessment]
  Offline --> LocalSnapshot[Local immutable snapshot and media]
  LocalSnapshot --> Outbox[(Durable SQLite-backed outbox)]
  Outbox --> Sync[Retrying synchronization worker]
  Sync --> Ingest[FastAPI sync routes]
  Ingest --> Central
  Ingest --> Attachments
  Central --> Queue[Clinician review queue]
  Attachments --> Queue
  Queue --> Decision[Separate agree or override review]
  Decision -. does not mutate .-> CentralSnapshot
```

The edge workflow saves its local assessment snapshot and media before synchronization. The sync routes persist that submitted evidence centrally; they do not rerun the assessment. Clinician review is stored separately and does not modify the snapshot.
