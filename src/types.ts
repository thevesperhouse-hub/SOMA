export interface TrainConfig {
  project_name: string;
  arch: string; // model family id (see /api/families)
  base_model: string;
  dataset_dir: string;
  instance_token: string;
  output_dir: string;
  resolution: number;
  rank: number;
  alpha: number;
  learning_rate: number;
  max_steps: number;
  batch_size: number;
  lr_warmup_ratio: number;
  min_snr_gamma: number;
  caption_dropout: number;
  reg_dataset_dir: string;
  prior_loss_weight: number;
  class_prompt: string;
  gradient_checkpointing: boolean;
  mixed_precision: "bf16" | "fp16" | "fp32";
  precision: string; // bf16 | int8 | nf4 (weight quantization)
  sample_every: number;
  sample_prompt: string;
  seed: number;
  simulate: boolean;
}

export type TrainState =
  | "idle"
  | "starting"
  | "training"
  | "sampling"
  | "done"
  | "stopped"
  | "error"
  | "captioning"
  | "done_caption";

export interface CaptionConfig {
  dataset_dir: string;
  instance_token: string;
  model_id: string;
  prompt: string;
  max_new_tokens: number;
  prepend_token: boolean;
  overwrite: boolean;
  output_dir: string;
}

export interface DatasetImage {
  path: string;
  name: string;
  caption: string;
}

export interface CaptionEvent {
  type: "caption";
  index: number;
  total: number;
  file: string;
  text: string;
  skipped: boolean;
}

export interface StepEvent {
  type: "step";
  step: number;
  total_steps: number;
  loss: number;
  lr: number;
  secs: number;
}

export interface SampleEvent {
  type: "sample";
  step: number;
  total_steps: number;
  placeholder: boolean;
  image?: string;
  seed?: number;
  prompt: string;
  sharpness: number;
}

export interface StatusEvent {
  type: "status";
  state: TrainState;
  total_steps?: number;
  step?: number;
  message?: string;
}

export interface LogEvent {
  type: "log";
  level: "info" | "warn" | "error";
  message: string;
}

export interface CaptionModelEvent {
  type: "caption_model";
  state: "downloading" | "loading" | "ready";
  percent?: number;
  mb?: number;
  total_mb?: number;
}

export type TrainEvent =
  | StepEvent
  | SampleEvent
  | StatusEvent
  | LogEvent
  | CaptionEvent
  | CaptionModelEvent;

export interface Sample {
  step: number;
  total: number;
  placeholder: boolean;
  image?: string;
  seed: number;
  sharpness: number;
}
