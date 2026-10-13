// Research page: topic and status filters, search, live counts, copy citation.
// Without JavaScript the full list still renders; the filter bar stays hidden.
(function () {
  "use strict";
  var bar = document.getElementById("research-filters");
  var papers = Array.prototype.slice.call(document.querySelectorAll("article.pub"));
  if (!bar || !papers.length) return;
  bar.hidden = false;

  var chips = Array.prototype.slice.call(bar.querySelectorAll(".chip"));
  var search = document.getElementById("paper-search");
  var count = document.getElementById("paper-count");
  var empty = document.getElementById("research-empty");

  function picked(group) {
    return chips.filter(function (c) { return c.dataset.group === group && c.getAttribute("aria-pressed") === "true"; })
      .map(function (c) { return c.dataset.k; });
  }
  function text(p) { return p.textContent.replace(/\s+/g, " ").toLowerCase(); }
  function matches(p, ignore) {
    var topics = picked("topic"), statuses = picked("status"), term = search.value.trim().toLowerCase();
    var own = (p.dataset.topics || "").split(" ");
    if (ignore !== "topic" && topics.length && !topics.some(function (t) { return own.indexOf(t) > -1; })) return false;
    if (ignore !== "status" && statuses.length && statuses.indexOf(p.dataset.status) < 0) return false;
    if (term && text(p.querySelector(".pub-title")).indexOf(term) < 0 && text(p.querySelector(".pub-row")).indexOf(term) < 0) return false;
    return true;
  }
  function update() {
    var shown = 0;
    papers.forEach(function (p) { var ok = matches(p); p.hidden = !ok; if (ok) shown++; });
    document.querySelectorAll("[data-section]").forEach(function (s) {
      s.hidden = !s.querySelector("article.pub:not([hidden])");
    });
    chips.forEach(function (c) {
      var n = papers.filter(function (p) {
        var hit = c.dataset.group === "topic" ? (p.dataset.topics || "").split(" ").indexOf(c.dataset.k) > -1 : p.dataset.status === c.dataset.k;
        return hit && matches(p, c.dataset.group);
      }).length;
      c.querySelector(".n").textContent = n;
    });
    count.textContent = "Showing " + shown + " of " + papers.length + " papers";
    empty.hidden = shown > 0;
    empty.textContent = /redact|classified|secret/.test(search.value.toLowerCase())
      ? "Nothing to see here. Move along."
      : "No papers match these filters. Clear a filter or the search to see more.";
  }

  chips.forEach(function (c) {
    c.addEventListener("click", function () {
      c.setAttribute("aria-pressed", String(c.getAttribute("aria-pressed") !== "true"));
      update();
    });
  });
  search.addEventListener("input", update);
  document.getElementById("clear-filters").addEventListener("click", function () {
    chips.forEach(function (c) { c.setAttribute("aria-pressed", "false"); });
    search.value = "";
    update();
  });
  update();

  // "Copy link" copies a URL that opens the page at this paper. Without JavaScript it is a plain anchor.
  document.querySelectorAll("a.copy-link").forEach(function (a) {
    a.addEventListener("click", function (e) {
      e.preventDefault();
      // a.hash, not the href: Quarto rewrites in-page hrefs into full URLs
      var hash = a.hash, url = location.origin + location.pathname + hash;
      var toast = a.closest("article").querySelector(".pub-toast");
      history.replaceState(null, "", hash);
      function fallback() { toast.textContent = "Copy this link: " + url; }
      try {
        navigator.clipboard.writeText(url).then(function () { toast.textContent = "Link to this paper copied"; }, fallback);
      } catch (err) { fallback(); }
    });
  });

  document.querySelectorAll(".cite-btn").forEach(function (b) {
    var art = b.closest("article"), bib = art.querySelector("pre.bibtex"), toast = art.querySelector(".pub-toast");
    b.hidden = false;
    b.addEventListener("click", function () {
      function fallback() { bib.hidden = false; toast.textContent = "Clipboard unavailable. The BibTeX is shown below."; }
      try {
        navigator.clipboard.writeText(bib.textContent).then(function () { toast.textContent = "BibTeX copied"; }, fallback);
      } catch (e) { fallback(); }
    });
  });
})();
