const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const DATA = path.join(ROOT, "data");
const BUILD = DATA;
const SIBLING_TOOLS = process.env.TOOLS_PATH || path.resolve(ROOT, "..", "charybdis-tools");

function ensureBuildDir() {
  if (!fs.existsSync(DATA)) fs.mkdirSync(DATA, { recursive: true });
}

function readBuild(name) {
  return JSON.parse(fs.readFileSync(path.join(DATA, name), "utf-8"));
}

function writeBuild(name, obj) {
  ensureBuildDir();
  fs.writeFileSync(path.join(DATA, name), JSON.stringify(obj, null, 2), "utf-8");
}

module.exports = { ROOT, BUILD, DATA, SIBLING_TOOLS, readBuild, writeBuild, ensureBuildDir };
