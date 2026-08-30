// The running build's version, substituted by Vite from package.json at build and dev-server
// start; see scripts/app-version.mjs for the single place it is read. The About panel and the
// update check therefore always agree, and an applied update moves the number with the code.
//
// The `typeof` guard is what keeps this importable outside a Vite build, where tests load it
// directly under Node and the constant was never substituted.
export const APP_VERSION = typeof __XIRAI_APP_VERSION__ === "string" ? __XIRAI_APP_VERSION__ : "0.0.0";
