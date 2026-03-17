/**
 * Citation hover tooltips for Computational Imaging course notes.
 *
 * sphinxcontrib-bibtex already places the full formatted reference text in
 * the `title` attribute of every in-text citation link. This script replaces
 * the browser's plain tooltip with a styled floating card.
 */
document.addEventListener("DOMContentLoaded", function () {

  /* ── Create tooltip element ─────────────────────────────────── */
  var tip = document.createElement("div");
  tip.id = "ci-cite-tooltip";
  document.body.appendChild(tip);

  var hideTimer = null;

  /* ── Find citation links ────────────────────────────────────── */
  // Citation <a> elements link to the references page and carry a `title`
  // attribute with the formatted reference text.
  var links = document.querySelectorAll("a.reference.internal[title]");

  links.forEach(function (a) {
    var href = a.getAttribute("href") || "";
    var title = a.getAttribute("title") || "";

    // Only act on links that target the references page and have content.
    if (!href.includes("reference") || !title) return;

    // Move the text into a data attribute and strip the browser tooltip.
    a.setAttribute("data-cite-text", title);
    a.removeAttribute("title");

    /* show ─────────────────────────────────────────────────────── */
    a.addEventListener("mouseenter", function (e) {
      clearTimeout(hideTimer);
      tip.innerHTML = "<p>" + escapeHtml(a.dataset.citeText) + "</p>";
      tip.classList.add("visible");
      positionTip(e);
    });

    /* track ────────────────────────────────────────────────────── */
    a.addEventListener("mousemove", positionTip);

    /* hide ─────────────────────────────────────────────────────── */
    a.addEventListener("mouseleave", function () {
      hideTimer = setTimeout(function () {
        tip.classList.remove("visible");
      }, 120);
    });
  });

  /* ── Position the tooltip near the cursor ───────────────────── */
  function positionTip(e) {
    var margin = 18;
    var vw = window.innerWidth;
    var vh = window.innerHeight;

    // Reset position so offsetWidth/Height are accurate
    tip.style.left = "0px";
    tip.style.top  = "0px";

    var tw = tip.offsetWidth;
    var th = tip.offsetHeight;

    var x = e.clientX + margin;
    var y = e.clientY + margin;

    // Flip left if overflowing right edge
    if (x + tw > vw - margin) x = e.clientX - tw - margin;
    // Flip up if overflowing bottom
    if (y + th > vh - margin) y = e.clientY - th - margin;

    tip.style.left = x + "px";
    tip.style.top  = y + "px";
  }

  /* ── Minimal HTML escaper ───────────────────────────────────── */
  function escapeHtml(str) {
    return str
      .replace(/&/g,  "&amp;")
      .replace(/</g,  "&lt;")
      .replace(/>/g,  "&gt;")
      .replace(/"/g,  "&quot;");
  }
});
