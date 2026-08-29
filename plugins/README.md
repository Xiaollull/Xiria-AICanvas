# XiriaCanvas AI plugin port

This folder is the plugin extension port. It is **user-owned**: the application never writes here,
the offline updater never replaces or deletes anything here, and release packaging never includes
plugin folders.

## Layout

```text
plugins/
  README.md            <- this file; it is a root file and is never treated as a plugin
  <plugin-id>/
    plugin.json        <- the only file the host reads (UTF-8 without BOM, max 64 KiB)
    ...                <- your program code and assets, never read or executed by this version
```

The folder name must be exactly equal to the manifest `id`: lowercase ASCII kebab-case, 3–64
characters.

## What the host does with this folder

Discovery, validation, a registry, and an enable/disable preference — nothing else.

`GET /api/plugins` reports what was found, and **Settings → 插件扩展** lists it with one switch per
plugin. Plugin code is **never** imported, required, executed, spawned, bundled, or served over
HTTP, and `/plugins/*` returns `404` on both the dev and preview servers. A folder full of
executable-looking code stays completely inert.

The switch is **not an execution grant**. It records your choice in `state-cache/plugins.json` and
changes nothing else. New plugins are off by default and must be turned on explicitly.

## Documentation

- `.Structure/12-PLUGIN-ARCHITECTURE-API.md` — architecture, manifest schema, registry API, security model.
- `.Structure/13-PLUGIN-DEVELOPMENT-GUIDELINES-ZH.md` — 中文插件开发使用准则与代码编写规范。

To turn a plugin off, use the switch in Settings. To remove it from the list entirely, move its
folder out of `plugins/`.
