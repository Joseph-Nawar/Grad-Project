from pydantic import BaseModel, Field


class AcuteStrokeSymptoms(BaseModel):
    """
    Structured acute stroke symptom inputs.

    This schema is designed for a research prototype and should not be
    interpreted as a validated diagnostic instrument.
    """

    face_drooping: bool = Field(
        default=False,
        description="Whether one side of the face appears drooped or asymmetric.",
    )

    arm_weakness: bool = Field(
        default=False,
        description="Whether one arm is weak, numb, or drifts downward.",
    )

    speech_difficulty: bool = Field(
        default=False,
        description="Whether speech is slurred, absent, confused, or hard to understand.",
    )

    balance_or_coordination_loss: bool = Field(
        default=False,
        description="Sudden trouble walking, dizziness, loss of balance, or coordination.",
    )

    vision_disturbance: bool = Field(
        default=False,
        description="Sudden blurred vision, double vision, or vision loss.",
    )

    sudden_severe_headache: bool = Field(
        default=False,
        description="Sudden severe headache with no known cause.",
    )

    confusion_or_understanding_difficulty: bool = Field(
        default=False,
        description="Sudden confusion or difficulty understanding speech.",
    )

    symptom_onset_minutes: int | None = Field(
        default=None,
        ge=0,
        description="Minutes since symptom onset, if known.",
    )

    symptoms_resolved: bool = Field(
        default=False,
        description="Whether symptoms have resolved by the time of assessment.",
    )