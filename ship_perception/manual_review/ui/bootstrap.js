import { mount } from "/workbench.js";
const root = document.querySelector("[data-mhr-mount]");
const event = document.querySelector("[data-mhr-event]");
const show = label => { if (event) event.textContent = label; };
const token = location.hash.slice(1) || sessionStorage.getItem("mhr-r2-token");
if (location.hash) {
  sessionStorage.setItem("mhr-r2-token", token);
  history.replaceState(null, "", location.pathname);
}
const workbench = mount(root, {
  token,
  onReviewChange: view => show("场景 " + view.scene_state),
  onPreview: candidate => show("预览 " + (candidate.match_state || candidate.execution_state)),
  onCommit: revision => show("已保存 " + (revision.state || revision.scene_state)),
  onError: error => show("请求失败 " + error.message),
});
workbench.openFrame();
