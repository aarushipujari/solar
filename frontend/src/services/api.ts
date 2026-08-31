import axios from "axios";

const API_BASE = "http://localhost:8000";

export interface PredictResponse {
  observation_time: string;
  forecast_window: {
    start_utc: string;
    end_utc: string;
  };
  target_active_region: string;
  data_mode: string;
  mx_probability_24h: number;
  mx_probability_48h: number;
  calibrated_probability: number;
  model_confidence: number;
  predicted_class: string;
  multiclass_distribution: {
    Quiet_B: number;
    C_Class: number;
    M_Class: number;
    X_Class: number;
  };
  estimated_peak_flux: string;
  risk_level: string;
  explanation_available: boolean;
  optical_proxies: {
    peak_intensity: number;
    mean_intensity: number;
    total_flux_proxy: number;
    max_gradient: number;
    mean_gradient: number;
    active_pixel_count: number;
    complexity_index: number;
  };
  mitigation_directives: Array<{
    sector: string;
    status: string;
    directive: string;
    level: string;
  }>;
}

export interface GradCamFrame {
  step: string;
  patch_base64: string;
  gradcam_base64: string;
  peak_attention_score: number;
}

export interface GradCamResponse {
  attribution_note: string;
  frames: GradCamFrame[];
}

export interface SolarChannel {
  id: string;
  name: string;
  description: string;
  image_base64: string;
}

export interface SolarChannelsResponse {
  full_disk: string;
  channels: SolarChannel[];
}

export const fetchPrediction = async (scenario_id: string = "AR3664_Impending_X_Flare"): Promise<PredictResponse> => {
  const res = await axios.post<PredictResponse>(`${API_BASE}/predict`, {
    scenario_id,
    data_mode: "DEMO"
  });
  return res.data;
};

export const fetchGradCam = async (scenario_id: string = "AR3664_Impending_X_Flare"): Promise<GradCamResponse> => {
  const res = await axios.get<GradCamResponse>(`${API_BASE}/api/gradcam?scenario_id=${scenario_id}`);
  return res.data;
};

export const fetchSolarChannels = async (scenario_id: string = "AR3664_Impending_X_Flare"): Promise<SolarChannelsResponse> => {
  const res = await axios.get<SolarChannelsResponse>(`${API_BASE}/api/solar-channels?scenario_id=${scenario_id}`);
  return res.data;
};

export const fetchBulletin = async (): Promise<string> => {
  const res = await axios.get<string>(`${API_BASE}/bulletin`, { responseType: "text" });
  return res.data;
};

export const fetchHealth = async () => {
  const res = await axios.get(`${API_BASE}/health`);
  return res.data;
};