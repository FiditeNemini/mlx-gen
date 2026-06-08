from __future__ import annotations

from dataclasses import dataclass

STATUS_PASS = "PASS"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAIL = "FAIL"
STATUS_STALE = "STALE"
STATUS_NA = "N/A"
STATUS_UNREVIEWED = "UNREVIEWED"
STATUS_NOT_AVAILABLE = "NOT_AVAILABLE"

_STATUS_RANK = {
    STATUS_FAIL: 0,
    STATUS_STALE: 1,
    STATUS_UNREVIEWED: 2,
    STATUS_PARTIAL: 3,
    STATUS_PASS: 4,
    STATUS_NA: 5,
}

I2I_EDIT_5X4_PROFILE_ID = "i2i_edit_5x4_2026_06_05"
REFRAME_OUTPAINT_PROFILE_ID = "reframe_outpaint_2026_06_08"

CANONICAL_SOURCE = "docs/assets/examples/spaceship-snow/01_t2i_spaceship_snow.png"
QWEN2511_PARITY_DIR = "docs/assets/validation/qwen-edit-2511-parity-2026-06-06"
REFRAME_OUTPAINT_DIR = "docs/assets/validation/reframe-outpaint-2026-06-08"
REFRAME_OUTPAINT_SOURCE = f"{REFRAME_OUTPAINT_DIR}/source-b-cropped-starship.png"


@dataclass(frozen=True)
class ValidationRecord:
    profile_id: str
    model: str
    family: str
    package_variant: str
    step: str
    step_label: str
    public_task: str
    mode: str
    status: str
    artifact_path: str | None
    source_images: tuple[str, ...]
    prompt: str
    reviewer_notes: str
    evidence_date: str = "2026-06-05"
    evidence_type: str = "manual_visual_review"

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "family": self.family,
            "package_variant": self.package_variant,
            "step": self.step,
            "step_label": self.step_label,
            "public_task": self.public_task,
            "mode": self.mode,
            "status": self.status,
            "artifact_path": self.artifact_path,
            "source_images": list(self.source_images),
            "prompt": self.prompt,
            "reviewer_notes": self.reviewer_notes,
            "evidence_date": self.evidence_date,
            "evidence_type": self.evidence_type,
        }


@dataclass(frozen=True)
class ValidationProfile:
    id: str
    title: str
    canonical_source: str
    description: str
    records: tuple[ValidationRecord, ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "canonical_source": self.canonical_source,
            "description": self.description,
            "records": [record.to_dict() for record in self.records],
        }


@dataclass(frozen=True)
class ModelValidation:
    profile_id: str
    model: str
    status: str
    records: tuple[ValidationRecord, ...]

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "model": self.model,
            "status": self.status,
            "records": [record.to_dict() for record in self.records],
        }


PROMPTS = {
    "B": (
        "Make this same spaceship in the snow look like polished cinematic science-fiction concept art at blue hour. "
        "Preserve the exact camera angle, ship position, snowy canyon, and overall layout. Sharpen hull panels and "
        "add cold blue shadows; no crash, no damage."
    ),
    "C": (
        "Edit the source into the same spaceship after a hard landing in the snow. Preserve the same camera angle, "
        "spaceship identity, rear engines, canyon cliffs, and wide framing. The ship must remain solid and sharp, "
        "but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, "
        "and a thin smoke plume. No blur, no mesh, no dissolve."
    ),
    "D": (
        "Turn the source into a clean graphite pencil sketch of the same hard-landed spaceship scene. Preserve the "
        "spaceship identity, snowy canyon layout, tilted hull, bent landing struts, disturbed snow, debris, and smoke "
        "plume. White paper background, precise line art, no color fill, no blur."
    ),
    "E": (
        "Use the first image as the pencil crash structure and the second image as the cinematic lighting and color "
        "reference. Produce one coherent image of the same hard-landed spaceship in the snow: graphite sketch lines "
        "with subtle cinematic blue-hour shading, stable canyon layout, solid spaceship, visible crash debris, no blur, "
        "no text."
    ),
}

STEP_LABELS = {
    "B": "cinematic reference",
    "C": "crash from source",
    "D": "pencil sketch",
    "E": "multi-reference composition",
}

_MATRIX_DIR = "docs/assets/validation/i2i-edit-5x4-2026-06-05"


def list_validation_profiles() -> tuple[ValidationProfile, ...]:
    return (_i2i_edit_profile(), _reframe_outpaint_profile())


def get_validation_profile(profile_id: str = I2I_EDIT_5X4_PROFILE_ID) -> ValidationProfile:
    for profile in list_validation_profiles():
        if profile.id == profile_id:
            return profile
    raise KeyError(f"Unknown validation profile {profile_id!r}.")


def get_model_validation(model: str, profile_id: str = I2I_EDIT_5X4_PROFILE_ID) -> ModelValidation:
    profile = get_validation_profile(profile_id)
    model_keys = _candidate_model_keys(model)
    records = tuple(record for record in profile.records if _normalize_model_key(record.model) in model_keys)
    return ModelValidation(
        profile_id=profile.id,
        model=model,
        status=_aggregate_status(records),
        records=records,
    )


def _aggregate_status(records: tuple[ValidationRecord, ...]) -> str:
    if not records:
        return STATUS_NOT_AVAILABLE
    return min((record.status for record in records), key=lambda status: _STATUS_RANK.get(status, -1))


def _normalize_model_key(model: str) -> str:
    return model.lower().replace("_", "-").rstrip("/")


def _candidate_model_keys(model: str) -> set[str]:
    keys = {_normalize_model_key(model)}
    try:
        from mflux.models.common.config import ModelConfig
        from mflux.utils.exceptions import ModelConfigError

        model_config = ModelConfig.from_name(model)
    except (ModelConfigError, ValueError):
        return keys

    keys.add(_normalize_model_key(model_config.model_name))
    return keys


def _i2i_edit_profile() -> ValidationProfile:
    return ValidationProfile(
        id=I2I_EDIT_5X4_PROFILE_ID,
        title="I2I Edit 5x4 Spaceship Snow Validation",
        canonical_source=CANONICAL_SOURCE,
        description=(
            "Manual visual QA for the spaceship-in-snow I2I profile. The profile separates route support from "
            "release validation and records exact model/package status for edit-reference, latent-img2img, and "
            "multi-reference cells used by the 5x4 contact sheets."
        ),
        records=tuple(_records()),
    )


def _reframe_outpaint_profile() -> ValidationProfile:
    return ValidationProfile(
        id=REFRAME_OUTPAINT_PROFILE_ID,
        title="Reframe And Outpaint Spaceship Validation",
        canonical_source=REFRAME_OUTPAINT_SOURCE,
        description=(
            "Manual visual QA for single-image edit-reference reframe and canvas-guided outpaint. "
            "The profile covers upstream source models plus q8 and q4 MLX-Gen packages for Qwen "
            "Image Edit, Qwen Image Edit 2509/2511, and FLUX.2 Klein 4B/9B."
        ),
        records=tuple(_reframe_outpaint_records()),
    )


def _records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    records.extend(_fibo_records())
    records.extend(
        _flux2_records(
            family="FLUX.2 Klein 4B",
            source_model="black-forest-labs/FLUX.2-klein-4B",
            source_slug="flux2_klein_4b_source",
            q8_model="AbstractFramework/flux.2-klein-4b-8bit",
            q8_slug="flux2_klein_4b_8bit",
            q4_model="AbstractFramework/flux.2-klein-4b-4bit",
            q4_slug="flux2_klein_4b_4bit",
        )
    )
    records.extend(
        _flux2_records(
            family="FLUX.2 Klein 9B",
            source_model="black-forest-labs/FLUX.2-klein-9B",
            source_slug="flux2_klein_9b_source",
            q8_model="AbstractFramework/flux.2-klein-9b-8bit",
            q8_slug="flux2_klein_9b_8bit",
            q4_model="AbstractFramework/flux.2-klein-9b-4bit",
            q4_slug="flux2_klein_9b_4bit",
        )
    )
    records.extend(_qwen2509_records())
    records.extend(_qwen2511_records())
    return records


QWEN_REFRAME_PROMPT = (
    "Generatively reframe this close-up into a wider establishing shot. Reveal the entire futuristic "
    "silver starship in the snowy alien plain, including the nose, full hull, both engines, landing "
    "legs, surrounding snow, and icy cliffs. Keep the same starship identity and material. Do not "
    "crop the ship. Keep it sharp and centered in a coherent wide frame."
)

QWEN_OUTPAINT_PROMPT = (
    "Outpaint this close cropped starship image into a much wider realistic shot of the full "
    "spacecraft in the snowy canyon. Keep the existing central spacecraft surface consistent, and "
    "complete the missing nose, full hull, tail, engines, snow field, and ice cliffs in the newly "
    "added space. The entire ship must fit inside the final wide frame with empty snow visible "
    "around it. Preserve the same lighting and camera angle. No text, no frame, no border, no "
    "duplicate ship."
)

QWEN2511_OUTPAINT_PROMPT = (
    "Outpaint this close cropped image into a wider realistic snowy canyon shot while keeping the "
    "same compact pod-like silver starship design from the source. Complete the missing nose, "
    "rounded hull, short tail, twin round rear engines, snow field, and ice cliffs in the newly "
    "added space. The final ship must remain a compact rounded spacecraft, not an airplane, with no "
    "large wings. Preserve the same lighting and camera angle. No text, no frame, no border, no "
    "duplicate ship."
)

FLUX_REFRAME_PROMPT = (
    "Generatively reframe this close-up into a wider establishing shot. Reveal the entire "
    "futuristic silver starship in the snowy alien plain, including the nose, full hull, both "
    "engines, landing legs, surrounding snow, and icy cliffs. Keep the same starship identity and "
    "material. Do not crop the ship. Keep it sharp and centered in a coherent wide frame. No "
    "duplicated spacecraft, no text, no border."
)

FLUX9_REFRAME_PROMPT = (
    "Zoom out from the source image into a wider snowy canyon view while keeping the exact same "
    "visible spacecraft design: a smooth silver sci-fi hull seen from the side, pointed nose on "
    "the left, one large circular black side engine intake, rounded metal body, short landing legs, "
    "and snowy canyon background. Use the larger canvas to reveal the missing rear, tail, full "
    "hull, surrounding snow, and ice cliffs. Keep the original side-view camera angle. Do not "
    "redesign it as an airplane, do not add long wings, propellers, or a front-facing cockpit "
    "aircraft view. No duplicate ship, no text, no border."
)

FLUX_OUTPAINT_PROMPT = (
    "Outpaint this close cropped starship image into a much wider realistic shot of the full "
    "spacecraft in the snowy canyon. Keep the existing compact silver spacecraft consistent, "
    "complete the missing nose, rounded hull, short tail, twin round rear engines, snow field, and "
    "ice cliffs in the newly added space. The entire ship must fit inside the final wide frame. No "
    "duplicated spacecraft, no repeated mountains, no text, no border."
)


def _reframe_outpaint_records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    specs = (
        (
            "Qwen Image Edit",
            QWEN_REFRAME_PROMPT,
            QWEN_OUTPAINT_PROMPT,
            8201,
            8212,
            20,
            4,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "Qwen/Qwen-Image-Edit",
                    "qwen_edit_source_reframe_b.png",
                    "qwen_edit_source_outpaint_b_wide.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/qwen-image-edit-8bit",
                    "qwen_edit_q8_reframe_b.png",
                    "qwen_edit_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/qwen-image-edit-4bit",
                    "qwen_edit_q4_reframe_b.png",
                    "qwen_edit_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "Qwen Image Edit 2509",
            QWEN_REFRAME_PROMPT,
            QWEN_OUTPAINT_PROMPT,
            8301,
            8312,
            20,
            4,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "Qwen/Qwen-Image-Edit-2509",
                    "qwen2509_source_reframe_b.png",
                    "qwen2509_source_outpaint_b.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/qwen-image-edit-2509-8bit",
                    "qwen2509_q8_reframe_b.png",
                    "qwen2509_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/qwen-image-edit-2509-4bit",
                    "qwen2509_q4_reframe_b.png",
                    "qwen2509_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "Qwen Image Edit 2511",
            QWEN_REFRAME_PROMPT,
            QWEN2511_OUTPAINT_PROMPT,
            8401,
            8413,
            20,
            4,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "Qwen/Qwen-Image-Edit-2511",
                    "qwen2511_source_reframe_b.png",
                    "qwen2511_source_outpaint_b_retry_compact.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/qwen-image-edit-2511-8bit",
                    "qwen2511_q8_reframe_b.png",
                    "qwen2511_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/qwen-image-edit-2511-4bit",
                    "qwen2511_q4_reframe_b.png",
                    "qwen2511_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "FLUX.2 Klein 4B",
            FLUX_REFRAME_PROMPT,
            FLUX_OUTPAINT_PROMPT,
            8501,
            8512,
            16,
            1,
            "25%,50%,25%,50%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "black-forest-labs/FLUX.2-klein-4B",
                    "flux2_4b_source_reframe_b.png",
                    "flux2_4b_source_outpaint_b.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/flux.2-klein-4b-8bit",
                    "flux2_4b_q8_reframe_b.png",
                    "flux2_4b_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/flux.2-klein-4b-4bit",
                    "flux2_4b_q4_reframe_b.png",
                    "flux2_4b_q4_outpaint_b.png",
                ),
            ),
        ),
        (
            "FLUX.2 Klein 9B",
            FLUX9_REFRAME_PROMPT,
            FLUX_OUTPAINT_PROMPT,
            8604,
            8612,
            16,
            1,
            "25%,80%,25%,60%",
            "5%,80%,5%,60%",
            (
                (
                    "source",
                    "black-forest-labs/FLUX.2-klein-9B",
                    "flux2_9b_source_reframe_b_wide_anchors.png",
                    "flux2_9b_source_outpaint_b.png",
                ),
                (
                    "q8 prepared",
                    "AbstractFramework/flux.2-klein-9b-8bit",
                    "flux2_9b_q8_reframe_b.png",
                    "flux2_9b_q8_outpaint_b.png",
                ),
                (
                    "q4 prepared",
                    "AbstractFramework/flux.2-klein-9b-4bit",
                    "flux2_9b_q4_reframe_b.png",
                    "flux2_9b_q4_outpaint_b.png",
                ),
            ),
        ),
    )
    for (
        family,
        reframe_prompt,
        outpaint_prompt,
        reframe_seed,
        outpaint_seed,
        steps,
        guidance,
        reframe_padding,
        outpaint_padding,
        variants,
    ) in specs:
        for package_variant, model, reframe_output, outpaint_output in variants:
            records.append(
                _reframe_outpaint_record(
                    model=model,
                    family=family,
                    package_variant=package_variant,
                    step="RF",
                    step_label="generative reframe",
                    artifact_file=reframe_output,
                    prompt=reframe_prompt,
                    reviewer_notes=(
                        f"PASS at padding {reframe_padding}, seed {reframe_seed}, {steps} steps, guidance {guidance}."
                    ),
                )
            )
            records.append(
                _reframe_outpaint_record(
                    model=model,
                    family=family,
                    package_variant=package_variant,
                    step="OP",
                    step_label="canvas-guided outpaint",
                    artifact_file=outpaint_output,
                    prompt=outpaint_prompt,
                    reviewer_notes=(
                        f"PASS at padding {outpaint_padding}, seed {outpaint_seed}, {steps} steps, guidance {guidance}. "
                        "This is a generative canvas expansion, not a native masked fill/inpaint run."
                    ),
                )
            )
    return records


def _reframe_outpaint_record(
    *,
    model: str,
    family: str,
    package_variant: str,
    step: str,
    step_label: str,
    artifact_file: str,
    prompt: str,
    reviewer_notes: str,
) -> ValidationRecord:
    return ValidationRecord(
        profile_id=REFRAME_OUTPAINT_PROFILE_ID,
        model=model,
        family=family,
        package_variant=package_variant,
        step=step,
        step_label=step_label,
        public_task="image-to-image",
        mode="edit-reference",
        status=STATUS_PASS,
        artifact_path=f"{REFRAME_OUTPAINT_DIR}/{artifact_file}",
        source_images=(REFRAME_OUTPAINT_SOURCE,),
        prompt=prompt,
        reviewer_notes=reviewer_notes,
        evidence_date="2026-06-08",
    )


def _source_images(step: str, slug: str | None = None) -> tuple[str, ...]:
    if step != "E":
        return (CANONICAL_SOURCE,)
    if slug is None:
        raise ValueError("Multi-reference validation records require a source slug.")
    return (
        _reference_input_path(f"{slug}_d_pencil_crash.png"),
        _reference_input_path(f"{slug}_b_cinematic.png"),
    )


def _reference_input_path(file_name: str) -> str:
    return f"{_MATRIX_DIR}/reference-inputs/{file_name}"


def _record(
    *,
    model: str,
    family: str,
    package_variant: str,
    step: str,
    mode: str,
    status: str,
    artifact_path: str | None,
    source_images: tuple[str, ...],
    reviewer_notes: str,
    prompt: str | None = None,
    step_label: str | None = None,
    evidence_date: str = "2026-06-05",
) -> ValidationRecord:
    return ValidationRecord(
        profile_id=I2I_EDIT_5X4_PROFILE_ID,
        model=model,
        family=family,
        package_variant=package_variant,
        step=step,
        step_label=step_label or STEP_LABELS[step],
        public_task="image-to-image",
        mode=mode,
        status=status,
        artifact_path=artifact_path,
        source_images=source_images,
        prompt=prompt or PROMPTS[step],
        reviewer_notes=reviewer_notes,
        evidence_date=evidence_date,
    )


def _fibo_records() -> list[ValidationRecord]:
    artifact_path = _matrix_path("fibo-edit-variant-matrix.jpg")
    return [
        _record(
            model="briaai/Fibo-Edit",
            family="FIBO Edit",
            package_variant="source",
            step="D",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Source route preserves some spaceship structure but is overexposed and does not satisfy the crash/sketch edit.",
        ),
        _record(
            model="briaai/Fibo-Edit",
            family="FIBO Edit",
            package_variant="source",
            step="C",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Hard-landing edit collapses or loses the ship.",
        ),
        _record(
            model="models/fibo-edit-bf16",
            family="FIBO Edit",
            package_variant="BF16 prepared",
            step="D",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local BF16 prepared folder has the required final bias, but full validation still fails before release-quality output.",
        ),
        _record(
            model="models/fibo-edit-bf16",
            family="FIBO Edit",
            package_variant="BF16 prepared",
            step="C",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local BF16 prepared folder has the required final bias, but hard-landing validation still fails.",
        ),
        _record(
            model="models/fibo-edit-8bit",
            family="FIBO Edit",
            package_variant="q8 prepared",
            step="D",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local q8 prepared folder keeps sensitive paths unquantized, but full validation still fails before release-quality output.",
        ),
        _record(
            model="models/fibo-edit-8bit",
            family="FIBO Edit",
            package_variant="q8 prepared",
            step="C",
            mode="edit-reference",
            status=STATUS_FAIL,
            artifact_path=artifact_path,
            source_images=(CANONICAL_SOURCE,),
            reviewer_notes="Current local q8 prepared folder keeps sensitive paths unquantized, but hard-landing validation still fails.",
        ),
    ]


def _flux2_records(
    *,
    family: str,
    source_model: str,
    source_slug: str,
    q8_model: str,
    q8_slug: str,
    q4_model: str,
    q4_slug: str,
) -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    artifact_path = _matrix_path(f"{source_slug.rsplit('_', 1)[0].replace('_', '-')}-variant-matrix.jpg")
    specs = (
        ("source", source_model, source_slug),
        ("q8 prepared", q8_model, q8_slug),
        ("q4 prepared", q4_model, q4_slug),
    )
    for package_variant, model, _slug in specs:
        for step, suffix, mode, notes in (
            ("B", "cinematic", "latent-img2img", "Preserves spaceship and scene while adding cinematic polish."),
            (
                "D",
                "pencil_crash",
                "edit-reference",
                "Clean pencil hard-landing sketch with recognizable ship and crash/smoke cues.",
            ),
            (
                "C",
                "crash",
                "edit-reference",
                "Solid hard-landing edit with smoke/snow disruption and preserved spaceship identity.",
            ),
            (
                "E",
                "composition",
                "multi-reference",
                "Uses pencil/crash and cinematic references coherently; preserves hard-landing scene.",
            ),
        ):
            records.append(
                _record(
                    model=model,
                    family=family,
                    package_variant=package_variant,
                    step=step,
                    mode=mode,
                    status=STATUS_PASS,
                    artifact_path=artifact_path,
                    source_images=_source_images(step, _slug),
                    reviewer_notes=notes,
                )
            )
    return records


def _qwen2509_records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    artifact_path = _matrix_path("qwen-image-edit-2509-variant-matrix.jpg")
    specs = (
        ("source", "Qwen/Qwen-Image-Edit-2509", "qwen_edit_2509_source"),
        ("q8 prepared", "AbstractFramework/qwen-image-edit-2509-8bit", "qwen_edit_2509_8bit"),
        ("q4 prepared", "AbstractFramework/qwen-image-edit-2509-4bit", "qwen_edit_2509_4bit"),
    )
    for package_variant, model, _slug in specs:
        for step, suffix, status, notes in (
            ("B", "cinematic", STATUS_PASS, "Preserves spaceship and scene while adding cinematic polish."),
            (
                "D",
                "pencil_crash",
                STATUS_PASS,
                "Clean pencil hard-landing sketch with recognizable ship and crash/smoke cues.",
            ),
            (
                "C",
                "crash",
                STATUS_PASS,
                "Solid hard-landing edit with smoke/snow disruption and preserved spaceship identity.",
            ),
            (
                "E",
                "composition",
                STATUS_PARTIAL if package_variant == "q4 prepared" else STATUS_PASS,
                "Preserves sketch/crash structure but weakly applies the color reference."
                if package_variant == "q4 prepared"
                else "Uses pencil/crash and cinematic references coherently; preserves hard-landing scene.",
            ),
        ):
            records.append(
                _record(
                    model=model,
                    family="Qwen Image Edit 2509",
                    package_variant=package_variant,
                    step=step,
                    mode="multi-reference" if step == "E" else "edit-reference",
                    status=status,
                    artifact_path=artifact_path,
                    source_images=_source_images(step, _slug),
                    reviewer_notes=notes,
                )
            )
    return records


def _qwen2511_records() -> list[ValidationRecord]:
    records: list[ValidationRecord] = []
    artifact_path = f"{QWEN2511_PARITY_DIR}/qwen-image-edit-2511-source-q8-q4-parity.jpg"
    specs = (
        ("source", "Qwen/Qwen-Image-Edit-2511", "source"),
        ("q8 prepared", "AbstractFramework/qwen-image-edit-2511-8bit", "q8"),
        ("q4 prepared", "AbstractFramework/qwen-image-edit-2511-4bit", "q4"),
    )
    qwen2511_prompts = {
        "B": (
            "Convert the source image into a clean graphite pencil sketch on white paper. Preserve the same wide "
            "camera framing, the same spaceship shape, the icy canyon background, and the rear engines. Use thin "
            "gray pencil outlines with light hand shading only. The final image must clearly look like a hand "
            "drawn pencil sketch, not a blurred photo."
        ),
        "C": (
            "Edit the source into the same spaceship after a hard landing in the snow at dusk. Preserve the same "
            "wide camera angle, spaceship identity, rear engines, canyon cliffs, and framing. The ship must remain "
            "solid and sharp, but show a tilted hull, bent landing struts, broken ice chunks, disturbed snow, a "
            "shallow scrape trail, and a thin smoke plume. Use blue-hour dusk lighting. No blur, no mesh, no "
            "dissolve."
        ),
        "E": (
            "Use the first image as the graphite pencil sketch style reference and the second image as the "
            "hard-landing crash content reference. Produce one coherent wide image of the same spaceship crashed "
            "in the snowy canyon: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, "
            "broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity and canyon "
            "layout. No blur, no colored photo, no text."
        ),
    }
    qwen2511_q4_prompts = {
        "C": (
            "Wide establishing shot of the same spaceship after a hard landing in the snow at dusk. Preserve the "
            "original wide camera angle, full spaceship fully visible inside the frame, rear engines visible, "
            "canyon cliffs visible on both left and right sides, and snowy ground foreground. Show a tilted hull, "
            "bent landing struts, broken ice chunks, disturbed snow, a shallow scrape trail, and a thin smoke "
            "plume. Use blue-hour dusk lighting. Keep the ship solid and sharp."
        ),
        "E": (
            "Use the first image as the graphite pencil sketch style reference and the second image as the "
            "hard-landing crash content reference. Produce one coherent wide image of the same spaceship crashed "
            "in the snowy canyon: graphite pencil outlines on white paper, visible tilted hull, disturbed snow, "
            "broken ice chunks, scrape trail, and a thin smoke plume. Preserve the spaceship identity, full wide "
            "framing, and canyon layout. No blur, no colored photo, no close-up, no cropped spaceship, no text."
        ),
    }
    labels = {"B": "pencil sketch", "C": "crash from source", "E": "multi-reference composition"}
    notes = {
        "B": "Clean pencil sketch that preserves the source spaceship and canyon layout.",
        "C": "Hard-landing edit with visible dusk lighting, smoke, snow disruption, and preserved spaceship identity.",
        "E": "Composition uses the pencil style and hard-landing reference coherently.",
    }
    for package_variant, model, slug in specs:
        for step in ("B", "C", "E"):
            prompt = qwen2511_prompts[step]
            reviewer_notes = notes[step]
            if package_variant == "q4 prepared" and step in qwen2511_q4_prompts:
                prompt = qwen2511_q4_prompts[step]
                reviewer_notes += " The q4 row used an explicit crop-avoidance negative prompt; see the command log."
            records.append(
                _record(
                    model=model,
                    family="Qwen Image Edit 2511",
                    package_variant=package_variant,
                    step=step,
                    mode="multi-reference" if step == "E" else "edit-reference",
                    status=STATUS_PASS,
                    artifact_path=artifact_path,
                    source_images=_qwen2511_source_images(slug, step),
                    reviewer_notes=reviewer_notes,
                    prompt=prompt,
                    step_label=labels[step],
                    evidence_date="2026-06-06",
                )
            )
    return records


def _qwen2511_source_images(slug: str, step: str) -> tuple[str, ...]:
    if step != "E":
        return (CANONICAL_SOURCE,)
    return (
        f"{QWEN2511_PARITY_DIR}/qwen2511-{slug}-pencil.png",
        f"{QWEN2511_PARITY_DIR}/qwen2511-{slug}-crash.png",
    )


def _matrix_path(file_name: str) -> str:
    return f"{_MATRIX_DIR}/{file_name}"
