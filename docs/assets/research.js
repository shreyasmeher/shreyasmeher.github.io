// Research page: copy citation.
(function () {
  "use strict";
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
