// Site behaviour: redaction patterns, the Auto/Light/Dark switch, the home-page
// wordmark, and a few easter eggs. Everything degrades to a working static page.
(function () {
  "use strict";
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- Redaction pattern ----------
     Lines of text with words blacked out in runs. Any element with class "pattern"
     and a data-seed gets one, sized to the element. Marks carry classes
     (pw word, pr redacted, pe evidence) so the theme colours them. */
  function rng(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function redaction(W, H, seed, density) {
    var r = rng(seed), rate = 0.07 * density, prev = false, step = 15, s = "";
    for (var y = 10; y < H - 4; y += step) {
      var x = -r() * 40;
      while (x < W) {
        var w = 10 + Math.pow(r(), 1.6) * 72, red = r() < (prev ? 0.55 : rate);
        if (red) s += '<rect class="pr" x="' + x.toFixed(1) + '" y="' + (y - 5) + '" width="' + w.toFixed(1) + '" height="10"/>';
        else if (r() < 0.012) s += '<rect class="pe" x="' + x.toFixed(1) + '" y="' + (y - 2) + '" width="' + w.toFixed(1) + '" height="4"/>';
        else s += '<rect class="pw" x="' + x.toFixed(1) + '" y="' + (y - 1.5) + '" width="' + w.toFixed(1) + '" height="3"/>';
        prev = red;
        x += w + 7;
      }
      if (r() < 0.08) y += step;
    }
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + W + " " + H + '" aria-hidden="true" focusable="false">' + s + "</svg>";
  }

  var drawn = new WeakMap();
  function draw() {
    document.querySelectorAll(".pattern[data-seed]").forEach(function (el) {
      var W = Math.round(el.clientWidth), H = Math.round(el.clientHeight);
      if (!W || !H || drawn.get(el) === W) return;
      drawn.set(el, W);
      el.innerHTML = redaction(W, H, parseInt(el.dataset.seed, 10) || 1, parseFloat(el.dataset.density) || 1);
    });
  }
  draw();
  var resizeTimer;
  window.addEventListener("resize", function () { clearTimeout(resizeTimer); resizeTimer = setTimeout(draw, 150); });

  /* ---------- Theme switch: Auto -> Light -> Dark ----------
     Built on Quarto's own toggle. "quarto-color-scheme" in localStorage holds
     "default" (light) or "alternate" (dark); no key means follow the system. */
  var KEY = "quarto-color-scheme";
  var systemDark = window.matchMedia("(prefers-color-scheme: dark)");
  var LABEL = { auto: "Auto", light: "Light", dark: "Dark" };
  var NEXT = { auto: "light", light: "dark", dark: "auto" };
  var TITLE = {
    auto: "Theme follows your system. Click for light.",
    light: "Light theme. Click for dark.",
    dark: "Dark theme, for late-night reviewing. Click to follow your system."
  };

  function stored() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  function mode() { var v = stored(); return v === "alternate" ? "dark" : v === "default" ? "light" : "auto"; }
  function isDark() { return document.body.classList.contains("quarto-dark"); }

  function setMode(m) {
    var wantDark = m === "dark" || (m === "auto" && systemDark.matches);
    if (wantDark !== isDark() && window.quartoToggleColorScheme) window.quartoToggleColorScheme();
    try {
      if (m === "auto") localStorage.removeItem(KEY);
      else localStorage.setItem(KEY, m === "dark" ? "alternate" : "default");
    } catch (e) {}
    paintSwitches();
  }

  var switches = [];
  function paintSwitches() {
    var m = mode();
    switches.forEach(function (b) {
      b.dataset.mode = m;
      b.querySelector(".theme-label").textContent = LABEL[m];
      b.title = TITLE[m];
      b.setAttribute("aria-label", "Colour theme: " + LABEL[m] + ". Change theme");
    });
  }

  document.querySelectorAll(".quarto-color-scheme-toggle").forEach(function (old) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "theme-switch";
    b.innerHTML = '<span class="theme-dot" aria-hidden="true"></span><span class="theme-label"></span>';
    b.addEventListener("click", function () { setMode(NEXT[mode()]); });
    old.parentNode.insertBefore(b, old);
    switches.push(b);
  });
  paintSwitches();

  /* ---------- Home: the navbar wordmark appears once the big name scrolls away ---------- */
  var heroName = document.querySelector(".hero-name");
  if (heroName && "IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      document.body.classList.toggle("name-out", !entries[0].isIntersecting);
    }, { rootMargin: "-64px 0px 0px 0px" }).observe(heroName);
  } else if (heroName) {
    document.body.classList.add("name-out");
  }

  /* ---------- Easter eggs ---------- */

  // 1. Click a black bar to declassify it. What was under it comes from the research:
  //    censorship, shutdowns, executive power, parliaments, and the methods behind them.
  //    Bars hold about 13 characters at most, so phrases come in three lengths.
  var WORDS = [
    // short: the tools and the targets
    "VPN", "DNS", "ISP", "gag", "ban", "veto", "writ", "edict", "leak", "purge", "OONI", "Lumen",
    "J&K", "AMvB", "RvS", "§144", "§69A", "TOI", "quorum",
    // medium: what the papers are about
    "decree", "blackout", "takedown", "firewall", "throttled", "Kashmir", "ordinance", "oversight",
    "coalition", "dissent", "evasion", "scrutiny", "The Hague", "Lok Sabha", "autocracy", "shutdown",
    "Rihanna", "codebook", "delegation", "court order", "checks",
    // long: named findings and terms of art
    "framework law", "Section 144", "Section 69A", "Strong Control", "Weak Control", "Tweede Kamer",
    "attention trap", "press freedom", "state capacity", "delegated act", "Kamervragen", "legal basis",
    "kill switch", "backsliding", "Article 19", "ConfliBERT", "ConflLlama", "placebo dates", "evasive reply"
  ];
  var used = {};
  function pickWord(t, w) {
    // Prefer an unused phrase that fills most of the bar, then any unused one that fits.
    var fits = [], full = [];
    WORDS.forEach(function (word) {
      t.textContent = word;
      var len = t.getComputedTextLength();
      if (len <= w - 2) { fits.push(word); if (len >= 0.55 * w) full.push(word); }
    });
    if (!fits.length) return null;
    var fresh = function (list) { return list.filter(function (x) { return !used[x]; }); };
    var pool = fresh(full).length ? fresh(full) : fresh(fits);
    if (!pool.length) { used = {}; pool = full.length ? full : fits; }
    var word = pool[Math.floor(Math.random() * pool.length)];
    used[word] = true;
    return word;
  }
  function declassify(rect) {
    // Use only the visible part of bars that run off either edge of the pattern.
    var W = rect.ownerSVGElement.viewBox.baseVal.width;
    var x0 = Math.max(0, +rect.getAttribute("x")), x1 = Math.min(W, +rect.getAttribute("x") + +rect.getAttribute("width"));
    var x = x0, w = Math.max(0, x1 - x0);
    var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", (x + w / 2).toFixed(1));
    t.setAttribute("y", +rect.getAttribute("y") + 8.5);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("class", "pd");
    rect.after(t);
    var word = pickWord(t, w);
    // Too narrow for any phrase: this one stays classified.
    if (!word) { t.remove(); return false; }
    t.textContent = word;
    rect.remove();
    return true;
  }
  document.addEventListener("click", function (e) {
    var rect = e.target.closest && e.target.closest("svg rect.pr");
    if (rect && !declassify(rect)) toast("This one stays classified.");
  });

  // 2. The Konami code lifts every redaction on the page.
  var KONAMI = ["ArrowUp", "ArrowUp", "ArrowDown", "ArrowDown", "ArrowLeft", "ArrowRight", "ArrowLeft", "ArrowRight", "b", "a"];
  var pos = 0;
  document.addEventListener("keydown", function (e) {
    pos = e.key === KONAMI[pos] ? pos + 1 : (e.key === KONAMI[0] ? 1 : 0);
    if (pos < KONAMI.length) return;
    pos = 0;
    var bars = document.querySelectorAll("svg rect.pr"), lifted = 0;
    bars.forEach(function (r) { if (declassify(r)) lifted++; });
    var kept = bars.length - lifted;
    toast(!bars.length ? "Nothing left to declassify."
      : "Declassified. " + lifted + " redaction" + (lifted === 1 ? "" : "s") + " lifted" + (kept ? ", " + kept + " still classified." : "."));
  });

  function toast(msg) {
    var el = document.createElement("div");
    el.className = "egg-toast";
    el.setAttribute("role", "status");
    el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(function () { el.classList.add("out"); }, reduceMotion ? 3500 : 3200);
    setTimeout(function () { el.remove(); }, 4000);
  }

  // 3. For the people who open the console.
  if (window.console && console.log) {
    console.log("%cNothing is redacted in the source.", "font: 600 14px Georgia, serif; color: #284678;");
    console.log("Click a black bar to declassify it, or try the Konami code. Hiring? meher@essb.eur.nl");
  }
})();
