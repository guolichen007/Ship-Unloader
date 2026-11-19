export type XY = [number, number];
export type XYZ = [number, number, number | null];
export interface Geometry {
  xy: [XY, XY, XY, XY]; xyz: [XYZ, XYZ, XYZ, XYZ];
  corner_ids: [string, string, string, string];
  length_m: number; width_m: number;
  xy_complete: boolean; xyz_complete: boolean;
  dimension_definition: string;
}
export interface Safety { steel_certified: false; control_eligible: false }
export interface Target extends Safety {
  opening_id: string; vessel_id: string; geometry: Geometry; geometry_digest: string;
  state: string; grade: string; source_kind: "AUTO_BASELINE";
}
export interface Revision extends Safety {
  annotation_id: string; revision: number; parent: number; state: string; geometry: Geometry | null;
}
export interface View extends Safety {
  manifest: { view_id: string; frame_id: string; nx: number; ny: number; x0: number; y0: number; s: number };
  auto: { status: string; bundle_digest: string | null; targets: Record<string, Target> };
  heads: Record<string, Revision>; pending: string[]; scene_state: string; scene_digest: string;
}
export interface Preview extends Safety {
  annotation_id: string; view_id: string; request_id: string; draft_digest: string;
  candidate_id: string; geometry_digest: string; geometry: Geometry | null;
  match_state: string | null; execution_state: string; ownership: "UNVERIFIED"; reason: string;
}
export interface WorkbenchOptions {
  token: string;
  onReviewChange?: (view: View) => void;
  onPreview?: (candidate: Preview) => void;
  onCommit?: (revision: Revision | (Safety & {scene_state: string; scene_digest: string})) => void;
  onError?: (error: Error) => void;
}
export interface WorkbenchComponent {
  openFrame(): Promise<void>;
  setTarget(id: string): boolean;
  resize(): void;
  destroy(): void;
}
export function mount(host: HTMLElement, options: WorkbenchOptions): WorkbenchComponent;
