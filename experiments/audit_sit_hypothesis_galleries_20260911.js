/* Execute each gallery's own selection logic and check every linked PNG. */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const crypto = require('crypto');
const assert = require('assert');
const base = '/home/zhoushunyu/data/eqvae/experiments';
const resultPath = '/home/zhoushunyu/eqvae/docs/data/sit_fsg_ctrl_hypothesis_20260911/gallery_audit.json';

class Element {
  constructor(value = '') { this.value = value; this._html = ''; this._text = ''; this.options = []; }
  append(element) { this.options.push(element); if (this.options.length === 1) this.value = String(element.value); }
  addEventListener() {}
  set innerHTML(value) { this._html = String(value); this._text = ''; }
  get innerHTML() { return this._html; }
  set textContent(value) { this._text = String(value); this._html = ''; }
  get textContent() { return this._text; }
}

function gallery(relative, defaults) {
  const root = path.join(base, relative), file = path.join(root, 'gallery.html');
  const html = fs.readFileSync(file, 'utf8');
  const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(match => match[1]);
  assert.strictEqual(scripts.length, 1);
  const elements = Object.fromEntries(Object.entries(defaults).map(([id, value]) => [id, new Element(String(value))]));
  const context = vm.createContext({document: {
    querySelector(id) { assert(id in elements, `Unknown element ${id}`); return elements[id]; },
    createElement(tag) { assert.strictEqual(tag, 'option'); return new Element(); }
  }});
  vm.runInContext(scripts[0], context, {timeout: 10000});
  const seen = new Set(); let selections = 0, references = 0;
  return {
    choose(values, expectedMinimum = 1) {
      for (const [id, value] of Object.entries(values)) elements[id].value = String(value);
      vm.runInContext('draw()', context, {timeout: 10000});
      const body = Object.values(elements).map(element => element.innerHTML).join('\n');
      const paths = [...body.matchAll(/<img\b[^>]*\bsrc="([^"]+)"/g)].map(match => match[1]);
      assert(paths.length >= expectedMinimum, `${relative}: incomplete selection ${JSON.stringify(values)}`);
      assert(!body.includes('NaN') && !body.includes('undefined'));
      for (const image of paths) {
        references++;
        if (seen.has(image)) continue;
        const target = path.resolve(root, image);
        assert(target.startsWith(root + path.sep));
        const handle = fs.openSync(target, 'r');
        const header = Buffer.alloc(24);
        try { assert.strictEqual(fs.readSync(handle, header, 0, 24, 0), 24); }
        finally { fs.closeSync(handle); }
        assert.strictEqual(header.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', target);
        assert.strictEqual(header.readUInt32BE(16), 256, target);
        assert.strictEqual(header.readUInt32BE(20), 256, target);
        seen.add(image);
      }
      selections++;
    },
    finish(expectedImages) {
      assert.strictEqual(seen.size, expectedImages, relative);
      return {gallery: file, passed: true, selections, references, unique_images: seen.size,
        gallery_sha256: crypto.createHash('sha256').update(html).digest('hex')};
    }
  };
}

const small = gallery('sit_fsg_ctrl_hypothesis_20260911', {
  '#mode': 'trajectory', '#sample': 0, '#tailstep': 32, '#base': 'cfg_tuned', '#step': 24,
  '#table': '', '#note': ''
});
for (let sample = 0; sample < 200; sample++) {
  for (const time of [8, 16, 24, 32, 40, 48]) {
    small.choose({'#mode': 'trajectory', '#sample': sample, '#tailstep': time}, 27);
  }
}
for (let sample = 0; sample < 64; sample++) {
  for (const prefix of ['cfg_tuned', 'cfg_high']) {
    for (const time of [8, 24, 40]) {
      small.choose({'#mode': 'intervention', '#sample': sample, '#base': prefix, '#step': time}, 28);
    }
  }
}
small.choose({'#mode': 'intervention', '#sample': 199}, 0);
const results = [small.finish(23400 + 5376 * 2 + 64 * 2 * 3 * 2)];

const xl = gallery('sit_xl_fsg_ctrl_examples_20260911', {'#sample': 0, '#view': ''});
for (let sample = 0; sample < 16; sample++) xl.choose({'#sample': sample}, 60);
results.push(xl.finish(960));

if (!process.argv.includes('--skip-direct')) {
  const direct = gallery('sit_fsg_ctrl_hypothesis_20260911/golden_path_direct', {
    '#sample': 0, '#time': 24, '#objective': 'agreement', '#view': ''
  });
  for (let sample = 0; sample < 16; sample++) {
    for (const time of [8, 24, 40]) {
      for (const objective of ['agreement', 'write', 'local']) {
        direct.choose({'#sample': sample, '#time': time, '#objective': objective}, 30);
      }
    }
  }
  results.push(direct.finish(4032));
}
const result = {passed: true, native_javascript_executed: true, all_selector_combinations_checked: true,
  all_linked_image_paths_exist: true, png_dimensions: [256, 256], full_png_decode_performed: false,
  browser_layout_rendered: false, complete: results.length === 3, galleries: results};
fs.writeFileSync(resultPath, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result, null, 2));
