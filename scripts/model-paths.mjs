import path from "node:path";

export const defaultModelPaths = {
  checkpoints: { sd: "models/checkpoints/sd", illustrious: "models/checkpoints/illustrious" },
  loras: {
    sd: "models/loras/sd",
    illustrious: "models/loras/illustrious",
    anima: "models/loras/anima",
    flux: "models/loras/flux",
    flux2: "models/loras/flux2",
    krea2: "models/loras/krea2",
  },
  vae: "models/vae",
  diffusion_models: "models/diffusion_models",
  text_encoders: "models/text_encoders",
  embeddings: "models/embeddings",
  upscalers: "models/upscalers",
  configs: "models/configs",
  yolo: "models/yolo",
  background_removal: "models/background-removal",
};

export function mergeModelPaths(config = {}) {
  if (!config || typeof config !== "object" || Array.isArray(config)) throw new Error("模型目录配置根节点必须是对象");
  for (const key of ["checkpoints", "loras"]) {
    if (config[key] != null && (typeof config[key] !== "object" || Array.isArray(config[key]))) {
      throw new Error(`模型目录 ${key} 必须是对象`);
    }
  }
  return {
    ...defaultModelPaths,
    ...config,
    checkpoints: { ...defaultModelPaths.checkpoints, ...(config.checkpoints || {}) },
    loras: { ...defaultModelPaths.loras, ...(config.loras || {}) },
  };
}

export function configuredModelDirectory(config, configKey, projectRoot, engineKey) {
  const merged = mergeModelPaths(config);
  const configured = engineKey ? merged[configKey]?.[engineKey] : merged[configKey];
  if (typeof configured !== "string" || !configured.trim() || path.isAbsolute(configured)) {
    throw new Error(`模型目录 ${configKey}${engineKey ? `.${engineKey}` : ""} 必须是非空的项目相对路径`);
  }
  const modelsRoot = path.resolve(projectRoot, "models");
  const directory = path.resolve(projectRoot, configured.trim());
  const relative = path.relative(modelsRoot, directory);
  if (!relative || relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`模型目录 ${configKey}${engineKey ? `.${engineKey}` : ""} 必须位于项目 models 文件夹内`);
  }
  return directory;
}

export function configuredModelDirectories(config, projectRoot) {
  const directories = [];
  for (const engine of ["sd", "illustrious"]) directories.push(configuredModelDirectory(config, "checkpoints", projectRoot, engine));
  for (const engine of ["sd", "illustrious", "anima"]) directories.push(configuredModelDirectory(config, "loras", projectRoot, engine));
  for (const key of ["vae", "diffusion_models", "text_encoders", "embeddings", "upscalers", "configs", "yolo", "background_removal"]) {
    directories.push(configuredModelDirectory(config, key, projectRoot));
  }
  return [...new Set(directories)];
}

export const defaultLoraCategories = [
  { id: "character", label: "角色", directory: "character" },
  { id: "style", label: "风格", directory: "style" },
  { id: "concept", label: "概念", directory: "concept" },
  { id: "other", label: "其他", directory: "other" },
];

export function groupLoraModels(models, rootDirectory, projectRoot) {
  const categories = new Map(defaultLoraCategories.map((category) => [category.id, {
    ...category,
    directory: path.relative(projectRoot, path.join(rootDirectory, category.directory)).split(path.sep).join("/"),
    models: [],
  }]));
  const rootModels = [];

  for (const model of models) {
    const segments = model.value.split("/").filter(Boolean);
    if (segments.length < 2) {
      rootModels.push({ ...model, name: segments[0] || model.name });
      continue;
    }
    const folder = segments[0];
    const known = defaultLoraCategories.find((category) => category.directory === folder);
    const id = known?.id || `folder:${encodeURIComponent(folder)}`;
    if (!categories.has(id)) {
      categories.set(id, {
        id,
        label: folder,
        directory: path.relative(projectRoot, path.join(rootDirectory, folder)).split(path.sep).join("/"),
        models: [],
      });
    }
    categories.get(id).models.push({ ...model, name: segments.slice(1).join("/") });
  }

  if (rootModels.length) {
    categories.set("root", {
      id: "root",
      label: "根目录",
      directory: path.relative(projectRoot, rootDirectory).split(path.sep).join("/"),
      models: rootModels,
    });
  }
  return [...categories.values()].sort((first, second) => {
    const firstDefault = defaultLoraCategories.findIndex((category) => category.id === first.id);
    const secondDefault = defaultLoraCategories.findIndex((category) => category.id === second.id);
    if (firstDefault >= 0 || secondDefault >= 0) {
      if (firstDefault < 0) return 1;
      if (secondDefault < 0) return -1;
      return firstDefault - secondDefault;
    }
    if (first.id === "root") return -1;
    if (second.id === "root") return 1;
    return first.label.localeCompare(second.label, "zh-CN");
  });
}
