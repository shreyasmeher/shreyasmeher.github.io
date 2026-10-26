// Research page: copy link and copy citation.
(function () {
  "use strict";
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
