/* Shared portal behaviour for every mock variant.

   The portal is one editable sheet: a fixed roster of 50 students
   (shared/roster.js), one row each. Index, Student ID and Student Name are
   printed by the portal and cannot be edited; every remaining column is an
   input the encoder (human or automated) fills in. Rows are never added or
   removed, so the roster length is invariant.

   State is in memory only - no backend, no persistence across reloads. Typing
   in a cell stages a value; Save All Rows validates the changed rows and
   commits them into `records`, which is what the evaluation harness reads.

   A variant supplies:
     - <form id="grade-form"> wrapping the sheet
     - <table id="records-table"> whose <thead> carries the entire column spec:
       one <th data-key> per column, plus data-type / data-options / data-min /
       data-max / data-step / data-placeholder / data-required on the editable
       ones. A <th data-key> with no data-type is roster-owned and printed.
     - <tbody id="records-body">, filled in here.

   Column order, column labelling and which columns exist are therefore
   variant-owned. No behaviour below depends on a header's text - header text is
   read only to word validation messages. */

(function () {
  "use strict";

  var ID_KEY = "student_id";
  var NAME_KEY = "student_name";
  var INDEX_KEY = "index";

  var cols = [];       // column spec, read from <thead> at start-up
  var fillable = [];   // keys of the editable columns
  var rows = [];       // <tr> per record, index-aligned with `records`

  // Identity comes from the roster; the fillable keys are added once the
  // column spec is known. The array identity is stable - the harness holds it.
  var records = (window.ROSTER || []).map(function (r) {
    return { student_id: r.student_id, student_name: r.student_name };
  });

  function table() { return document.getElementById("records-table"); }
  function form() { return document.getElementById("grade-form"); }
  function scale() { return table().dataset.scale || "0-100"; }

  // Variants opt out with data-associate-headers="false", which leaves every
  // input with no accessible name at all - the column header above it is then
  // the only thing that identifies it.
  function associateHeaders() {
    return table().dataset.associateHeaders !== "false";
  }

  /* ---- column spec ---- */

  function readColumns() {
    var ths = table().querySelectorAll("thead th[data-key]");
    return Array.prototype.map.call(ths, function (th, n) {
      if (!th.id) th.id = "col-" + th.dataset.key + "-" + n;
      return {
        key: th.dataset.key,
        type: th.dataset.type || "",      // "" -> roster-owned, printed not edited
        // Message wording only. data-label lets a header carry a format hint
        // ("Grade 0-100") without that hint leaking into validation messages.
        label: th.dataset.label || th.textContent.trim(),
        headerId: th.id,
        placeholder: th.dataset.placeholder || "",
        // The control's name attribute. Deliberately tracks the column's
        // visible label rather than its data-key: a relabelled variant renames
        // its fields, as a real relabelled system would, so nothing downstream
        // can recover the original identity from the name.
        name: th.dataset.name || "",
        maxlength: th.dataset.maxlength,
        options: th.dataset.options ? th.dataset.options.split(",") : [],
        min: th.dataset.min,
        max: th.dataset.max,
        step: th.dataset.step,
        required: th.hasAttribute("data-required")
      };
    });
  }

  function editable(col) { return col.type !== ""; }

  /* ---- sheet construction ---- */

  function buildInput(col, i, nameCellId) {
    var el;

    if (col.type === "select") {
      el = document.createElement("select");
      el.appendChild(new Option("", ""));
      col.options.forEach(function (o) { el.appendChild(new Option(o, o)); });
    } else if (col.type === "textarea") {
      el = document.createElement("textarea");
      el.rows = 1;
    } else {
      el = document.createElement("input");
      el.type = col.type === "number" ? "number" : "text";
      if (col.min !== undefined) el.min = col.min;
      if (col.max !== undefined) el.max = col.max;
      if (col.step !== undefined) el.step = col.step;
    }

    if (col.placeholder && el.tagName !== "SELECT") el.placeholder = col.placeholder;
    // Every row in a column shares the name, the way a form posting an array
    // of records does. A per-row suffix would leave the scanner describing the
    // column as "Grade 0" and put a row index into a semantic feature.
    if (col.name) el.name = col.name;

    // A unique id per cell, built from the column's own field name and the
    // student's record key - the shape a server-rendered form produces when it
    // loops over rows ("grade_2021-10001"). It matters for the RPA comparison:
    // without it a selector built from the name attribute matches all fifty
    // rows at once, and one recorded action rewrites the whole sheet.
    //
    // The id follows the *field name*, so a variant that renames Grade to Final
    // Rating renames its ids too. That is what a real rename does, and it is
    // what separates a cosmetic reorder from a genuine schema change.
    if (col.name && records[i]) el.id = col.name + "_" + records[i][ID_KEY];
    if (col.maxlength && el.tagName !== "SELECT") el.maxLength = Number(col.maxlength);

    // The form is novalidate and validation runs in validate(), but the
    // attribute still has to be on the control: it is what a page scanner
    // reads, and a portal whose DOM understates its own requirements is
    // lying to anything that inspects it.
    if (col.required) el.required = true;

    el.dataset.key = col.key;
    el.dataset.row = String(i);

    // No <label for> is possible in a sheet: the accessible name is the column
    // header plus the row's student-name cell, e.g. "Grade Cruz, Isabel K.".
    if (associateHeaders()) {
      el.setAttribute("aria-labelledby", col.headerId + " " + nameCellId);
    }
    return el;
  }

  function buildSheet() {
    var body = document.getElementById("records-body");
    body.innerHTML = "";
    rows = [];

    records.forEach(function (rec, i) {
      var tr = document.createElement("tr");
      tr.dataset.row = String(i);

      var pick = document.createElement("td");
      pick.className = "pick";
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "row-select";
      cb.setAttribute("aria-label", "Select " + rec[NAME_KEY]);
      pick.appendChild(cb);
      tr.appendChild(pick);

      var nameCellId = "row-" + i + "-name";

      cols.forEach(function (col) {
        var td = document.createElement("td");
        td.dataset.key = col.key;

        if (!editable(col)) {
          td.className = "fixed";
          if (col.key === INDEX_KEY) {
            td.classList.add("num");
            td.textContent = String(i + 1);
          } else {
            td.textContent = rec[col.key] || "";
            if (col.key === NAME_KEY) td.id = nameCellId;
          }
        } else {
          td.appendChild(buildInput(col, i, nameCellId));
        }

        tr.appendChild(td);
      });

      body.appendChild(tr);
      rows.push(tr);
    });
  }

  /* ---- reading and writing ---- */

  // Cells carry data-key too, so the query names the controls explicitly.
  var CONTROLS = "input[data-key], select[data-key], textarea[data-key]";

  function inputsIn(i) { return rows[i].querySelectorAll(CONTROLS); }

  // The staged (on-screen) values of one row, identity included.
  function readRow(i) {
    var rec = {};
    rec[ID_KEY] = records[i][ID_KEY];
    rec[NAME_KEY] = records[i][NAME_KEY];
    inputsIn(i).forEach(function (el) { rec[el.dataset.key] = el.value.trim(); });
    return rec;
  }

  function writeRow(i) {
    inputsIn(i).forEach(function (el) {
      el.value = records[i][el.dataset.key] || "";
    });
  }

  function isDirty(i) {
    var staged = readRow(i);
    return fillable.some(function (k) { return staged[k] !== records[i][k]; });
  }

  function isTouched(staged) {
    return fillable.some(function (k) { return staged[k] !== ""; });
  }

  function findRow(studentId) {
    var key = String(studentId).trim().toLowerCase();
    for (var i = 0; i < records.length; i++) {
      if (records[i][ID_KEY].toLowerCase() === key) return i;
    }
    return -1;
  }

  /* ---- validation ---- */

  // Returns { <key>: <message> } for one row's staged values. A row is only
  // validated once the encoder has put something in it, so 49 untouched rows
  // never block a save of the 50th.
  function validate(staged) {
    var errs = {};

    cols.filter(editable).forEach(function (col) {
      var raw = staged[col.key];

      if (raw === "") {
        if (col.required) errs[col.key] = col.label + " is required.";
        return;
      }

      if (col.type === "number") {
        var n = Number(raw);
        if (Number.isNaN(n)) {
          errs[col.key] = col.label + " must be a number.";
          return;
        }
        var lo = col.min === undefined ? -Infinity : Number(col.min);
        var hi = col.max === undefined ? Infinity : Number(col.max);
        if (n < lo || n > hi) {
          errs[col.key] = col.label + " must be between " + col.min + " and " + col.max + ".";
        }
      } else if (col.type === "select" && col.options.indexOf(raw) === -1) {
        errs[col.key] = col.label + " must be one of: " + col.options.join(", ") + ".";
      }
    });

    return errs;
  }

  function markErrors(i, errs) {
    inputsIn(i).forEach(function (el) {
      var msg = errs[el.dataset.key];
      var td = el.parentNode;
      td.classList.toggle("invalid", Boolean(msg));
      if (msg) {
        el.setAttribute("aria-invalid", "true");
        el.title = msg;
      } else {
        el.removeAttribute("aria-invalid");
        el.removeAttribute("title");
      }
    });
  }

  /* ---- presentation ---- */

  function refreshRow(i) {
    var tr = rows[i];
    tr.classList.toggle("dirty", isDirty(i));
    inputsIn(i).forEach(function (el) {
      el.parentNode.classList.toggle("blank", el.value.trim() === "");
    });
  }

  function refreshAll() {
    records.forEach(function (_, i) { refreshRow(i); });
    updateCount();
  }

  function updateCount() {
    var counter = document.getElementById("roster-count");
    if (!counter) return;

    var done = records.filter(function (r) { return isTouched(r); }).length;
    var unsaved = rows.filter(function (_, i) { return isDirty(i); }).length;
    counter.textContent = done + " of " + records.length + " encoded" +
      (unsaved ? " - " + unsaved + " unsaved" : "");
  }

  function setStatus(msg, kind) {
    var el = document.getElementById("form-status");
    if (!el) return;
    el.textContent = msg || "";
    el.className = "status" + (kind ? " " + kind : "");
  }

  function clearFlash() {
    rows.forEach(function (tr) { tr.classList.remove("just-saved"); });
  }

  function plural(n, word) { return n + " " + word + (n === 1 ? "" : "s"); }

  /* ---- save / revert ---- */

  function saveAll() {
    var saved = 0;
    var failed = 0;
    var firstBad = -1;

    clearFlash();

    records.forEach(function (rec, i) {
      if (!isDirty(i)) { markErrors(i, {}); return; }

      var staged = readRow(i);
      var errs = validate(staged);
      markErrors(i, errs);

      if (Object.keys(errs).length) {
        failed++;
        if (firstBad === -1) firstBad = i;
        return;
      }

      fillable.forEach(function (k) { rec[k] = staged[k]; });
      rows[i].classList.add("just-saved");
      saved++;
    });

    refreshAll();

    if (!saved && !failed) {
      setStatus("No changes to save.", null);
    } else if (failed) {
      var msg = plural(failed, "row") + " could not be saved - see the highlighted cells.";
      setStatus((saved ? "Saved " + plural(saved, "row") + ". " : "") + msg, "error");
      var badCell = rows[firstBad].querySelector("td.invalid");
      var bad = badCell && badCell.querySelector(CONTROLS);
      if (bad) bad.focus();
    } else {
      setStatus("Saved " + plural(saved, "row") + ".", "ok");
    }
  }

  function revertAll() {
    clearFlash();
    records.forEach(function (_, i) {
      writeRow(i);
      markErrors(i, {});
    });
    refreshAll();
    setStatus("Unsaved edits discarded.", null);
  }

  /* ---- wiring ---- */

  window.addEventListener("DOMContentLoaded", function () {
    cols = readColumns();
    fillable = cols.filter(editable).map(function (c) { return c.key; });
    records.forEach(function (rec) {
      fillable.forEach(function (k) { if (!(k in rec)) rec[k] = ""; });
    });

    buildSheet();
    refreshAll();

    form().addEventListener("submit", function (e) {
      e.preventDefault();
      saveAll();
    });

    // Editing clears the previous outcome, so a stale message is never read as
    // the result of the current attempt.
    var body = document.getElementById("records-body");
    body.addEventListener("input", function (e) {
      var el = e.target;
      if (!el.dataset || el.dataset.row === undefined) return;
      var i = Number(el.dataset.row);
      rows[i].classList.remove("just-saved");
      refreshRow(i);
      updateCount();
      setStatus("", null);
    });

    var revertBtn = document.getElementById("revert-btn");
    if (revertBtn) revertBtn.addEventListener("click", revertAll);

    var selectAll = document.getElementById("select-all");
    if (selectAll) {
      selectAll.addEventListener("change", function () {
        body.querySelectorAll(".row-select").forEach(function (cb) {
          cb.checked = selectAll.checked;
        });
      });
    }
  });

  // Exposed for the evaluation harness: readback, row lookup, dry-run assertions.
  window.__portal = {
    records: records,
    read: readRow,                                    // staged values of row i
    find: findRow,
    row: function (i) { return records[i]; },         // committed values of row i
    columns: function () { return cols.slice(); },
    scale: scale,
    save: saveAll
  };
})();
