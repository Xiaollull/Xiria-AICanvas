// Session-only, dependency-free frontend state. Transport and preview URLs belong to the host.
export const WD14_CACHE_LIMIT = 30;
export const WD14_IMAGE_BYTES = 20 * 1024 * 1024;

export class Wd14Desk {
  constructor() {
    this.state = { entries: [], selectedId: null, message: "" };
    this.listeners = new Set();
    this.serial = 0;
    this.epoch = 0;
    this.activeRequest = null;
    this.alive = true;
  }
  getSnapshot = () => this.state;
  subscribe = (listener) => { this.listeners.add(listener); return () => this.listeners.delete(listener); };
  publish(patch) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((listener) => listener());
  }
  update(id, patch) {
    this.publish({ entries: this.state.entries.map((entry) => entry.id === id ? { ...entry, ...patch } : entry) });
  }
  activate() { this.alive = true; }
  dispose() { this.alive = false; this.invalidate(); }
  invalidate() {
    this.epoch++;
    const pending = this.activeRequest;
    this.activeRequest = null;
    pending?.controller.abort();
    if (pending && this.alive) this.update(pending.id, { status: "stopped", notice: "已停止等待；服务端当前模型计算可能仍会继续，完成后释放其自身 GPU 资源。" });
  }
  add(files) {
    const incoming = [...files];
    if (incoming.length !== 1) { this.publish({ message: "当前为单图工作区：每次仅添加一张。多张图片请逐张添加，或使用文件夹批量。" }); return null; }
    const file = incoming[0];
    if (!/^image\//.test(file.type) && !/\.(png|jpe?g|webp|bmp|gif|tiff?|avif)$/i.test(file.name)) { this.publish({ message: "请选择支持的图片文件" }); return null; }
    if (file.size > WD14_IMAGE_BYTES) { this.publish({ message: "图片超过 20 MiB，请选择更小的图片；当前图片与结果未改变。" }); return null; }
    if (this.state.entries.filter((entry) => entry.result).length >= WD14_CACHE_LIMIT) { this.publish({ message: "已达 30 项缓存上限，请先移除当前结果或清空缓存，再添加图片。现有结果未改变。" }); return null; }
    this.invalidate();
    const retained = this.state.entries.filter((entry) => entry.result);
    const entry = { id: `wd14-${++this.serial}`, file, result: null, text: "", params: null, attemptParams: null, status: "idle", error: "", notice: "" };
    this.publish({ entries: [...retained, entry], selectedId: entry.id, message: "" });
    return entry.id;
  }
  select(id) {
    if (this.state.entries.some((entry) => entry.id === id)) this.publish({ selectedId: id, message: "" });
  }
  edit(id, text) { this.update(id, { text }); }
  removeSelected() {
    if (this.activeRequest?.id === this.state.selectedId) this.invalidate();
    const entries = this.state.entries.filter((entry) => entry.id !== this.state.selectedId);
    this.publish({ entries, selectedId: entries.at(-1)?.id || null, message: "" });
  }
  clear() { this.invalidate(); this.publish({ entries: [], selectedId: null, message: "" }); }
  stop() { this.invalidate(); }
  async run(id, parameters, transport) {
    const entry = this.state.entries.find((row) => row.id === id);
    if (!entry || !this.alive) return;
    this.invalidate();
    const params = Object.freeze({ ...parameters });
    const ticket = { id, epoch: this.epoch, controller: new AbortController() };
    this.activeRequest = ticket;
    this.update(id, { status: "pending", attemptParams: params, error: "", notice: "正在反推；新图片会替换当前输入，完成结果保留在下方。" });
    const valid = () => this.alive && this.activeRequest === ticket && this.epoch === ticket.epoch && this.state.entries.some((row) => row.id === id);
    try {
      const result = await transport(entry.file, params, ticket.controller.signal);
      if (!valid()) return;
      this.activeRequest = null;
      this.update(id, { status: "done", result, params, text: result.tags.map((tag) => tag.name).join(", "), error: "", notice: "反推完成；可编辑 Prompt，或点击下方缩略图切换结果。" });
    } catch (failure) {
      if (!valid()) return;
      this.activeRequest = null;
      // Keep the latest edited text and last successful result on any rerun failure.
      this.update(id, { status: "error", error: failure.message || "反推失败，请重试", notice: entry.result ? "重新反推失败，原结果和编辑文本已保留。" : "反推失败，可调整参数后重试。" });
    }
  }
}
