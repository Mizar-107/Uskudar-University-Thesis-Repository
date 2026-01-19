# SN Computer Science Journal Submission Progress

**Target Journal:** SN Computer Science (Springer Nature)
**Journal URL:** https://link.springer.com/journal/42979
**Submission System:** Editorial Manager (https://www.editorialmanager.com/sncs/)

---

## Paper Information
- **Title:** Fine-tuned YOLO11 for Maritime Vessel Detection in the Bosphorus Strait: A Comparative Study
- **Authors:** Recep Ertugrul Eksi, Rowanda D. Ahmed
- **Affiliation:** Uskudar University, Istanbul, Turkey
- **Corresponding Author:** Recep Ertugrul Eksi (recepertugrul.eksi@st.uskudar.edu.tr)

---

## Submission Checklist

### Template Conversion
- [x] Download/implement Springer Nature sn-jnl template
- [x] Convert document class from `article` to `sn-jnl`
- [x] Reformat author/affiliation block to Springer format
- [x] ~~Add ORCID identifiers~~ (not available)
- [x] Update abstract format
- [x] Convert keywords format

### Required Sections
- [x] Title, Authors, Affiliations
- [x] Abstract
- [x] Keywords
- [x] Main content (Introduction through Conclusion)
- [x] Acknowledgements
- [x] **Declarations section** (Required by Springer)
  - [x] Funding statement: "No funding was received for conducting this study."
  - [x] Competing interests: "The authors declare that they have no conflict of interest."
  - [x] Data availability: Dataset available at Roboflow Universe (CC BY 4.0)
  - [x] Code availability: GitHub repository linked
  - [x] Author contributions: Detailed for both authors
  - [x] Ethics approval: Not applicable (no human subjects)
- [x] References

### Figures & Tables
- [x] Verify figure formats (PNG - accepted)
- [ ] Check figure resolution (min 300 DPI for photos) - verify before submission
- [x] Ensure table formatting compatibility (8 tables converted)

### Bibliography
- [x] Using sn-basic.bst (Springer Nature bibliography style)
- [x] 10 references included
- [x] DOIs included where available

### Final Checks
- [ ] Compile without errors (test with pdflatex)
- [ ] Proofread converted document
- [x] Prepare cover letter (cover_letter.tex)
- [ ] Identify suggested reviewers (3-5 recommended)

---

## Resolved Questions

1. **Funding:** No funding was received
2. **Author Contributions:**
   - Recep Ertugrul Eksi: Conceptualization, Methodology, Software, Investigation, Writing - Original Draft, Visualization
   - Rowanda D. Ahmed: Supervision, Writing - Review & Editing, Validation
3. **ORCID IDs:** Not available
4. **Code Availability:** https://github.com/Mizar-107/Uskudar-University-Thesis-Repository
5. **Ethics:** Not applicable (no human subjects, publicly available data)

---

## Progress Log

### Session: 2025-12-29
- [x] Initial assessment complete
- [x] Fetched SN Computer Science submission guidelines
- [x] Downloaded Springer Nature sn-jnl.cls template (v3.1, Dec 2024)
- [x] Downloaded sn-basic.bst bibliography style
- [x] Created main_springer.tex with full paper conversion
- [x] Added complete Declarations section
- [x] Updated all section cross-references to Springer format
- [x] Committed changes to repository

---

## File Structure

```
Paper - Computer Science Journal/
├── main.tex                 (original - csagh template, kept as backup)
├── main_springer.tex        (NEW - Springer Nature submission)
├── cover_letter.tex         (NEW - Cover letter for submission)
├── bibliography.bib         (10 references)
├── sn-jnl.cls              (Springer Nature class file v3.1)
├── sn-basic.bst            (Springer bibliography style)
├── csagh.sty               (original template - backup)
├── cs-agh.bst              (original bib style - backup)
├── moreverb.sty            (original support file)
├── figures/
│   ├── comparison_grid.png  (detection comparison visualization)
│   └── comparison_charts.png (performance metrics charts)
└── SN_Computer_Science_Submission.md (this tracking file)
```

---

## Next Steps for Submission

1. **Compile the paper** - Run pdflatex on main_springer.tex to generate PDF
2. **Review the PDF** - Check formatting, figures, tables, and references
3. **Write cover letter** - Brief letter to the editor explaining the paper's significance
4. **Suggest reviewers** - Identify 3-5 experts in maritime detection/YOLO/computer vision
5. **Submit via Editorial Manager** - https://www.editorialmanager.com/sncs/

---

## Suggested Reviewers (to be finalized)

Consider experts in:
- YOLO/object detection (Ultralytics community)
- Maritime surveillance systems
- Transfer learning for domain adaptation
- Computer vision applications in transportation

---

## Important Links

- **Journal Homepage:** https://link.springer.com/journal/42979
- **Submission Guidelines:** https://link.springer.com/journal/42979/submission-guidelines
- **Editorial Manager:** https://www.editorialmanager.com/sncs/
- **LaTeX Template Info:** https://www.springernature.com/gp/authors/campaigns/latex-author-support
- **Dataset:** https://universe.roboflow.com/bosphorus-vision-project/bogaz_v_1-o9ldr
- **Code Repository:** https://github.com/Mizar-107/Uskudar-University-Thesis-Repository
