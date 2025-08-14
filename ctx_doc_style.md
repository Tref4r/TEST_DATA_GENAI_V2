---
title: ctx_doc_style
---

# Markdown Documentation Style Guidelines

---

## Content Preservation Rule

<details>
<summary>Critical Principle for Applying Style to Existing Documents</summary>

---

**When applying this markdown style to existing documents, preserve all original content exactly. Only change structure, organization, and formatting.**

### What Must Be Preserved
- **All text content** - every word, sentence, and paragraph
- **All technical information** - code examples, API details, specifications
- **All data** - tables, lists, numbers, values, measurements
- **All links and references** - URLs, file paths, citations

### What Can Be Changed
- **Section organization** - reorganizing content into appropriate details blocks
- **Header hierarchy** - adjusting ## ### #### levels for consistency
- **Content grouping** - moving related content together
- **Bullet point formatting** - converting paragraphs to bullet points while preserving all information

---

</details>

---

## Document Structure

<details>
<summary>Required Organization and Hierarchy Rules</summary>

---

### Front Matter (Required)
```yaml
---
title: document_name_in_snake_case
---
```

### Section ## Rules (Main Sections)
- **Always use `---` separators** before and after ## sections
- **Topic dividers only** - no content directly under ## sections
- **Example**:
  ```markdown
  ---
  ## Main Section A
  ---
  ## Main Section B
  ---
  ```

### Section ### Rules (Subsections)
- **Must have at most one `<details><summary>` block** containing all content
- **NO content outside details blocks** at subsection level
- **NO separators between ### sections** at the same level
- **Example**:
  ```markdown
  ### Subsection A
  <details>
  <summary>Content description</summary>
  ---
  All content here
  ---
  </details>

  ### Subsection B
  <details>
  <summary>Content description</summary>
  ---
  All content here
  ---
  </details>
  ```

### Section #### Rules (Subsubsections)
- **Exist only within details blocks**
- **Always separated by `---` horizontal rules**
- **Example**:
  ```markdown
  <details>
  <summary>Content description</summary>
  ---
  
  #### First Subsubsection
  Content here
  
  ---
  
  #### Second Subsubsection
  Content here
  
  ---
  </details>
  ```

### Critical Separator Rules
- **`---` separators are ONLY used**:
  - Between ## level sections (main sections)
  - At start and end of details blocks
  - Between #### sections within details blocks
- **NEVER use `---` separators**:
  - Between ### level sections
  - Outside of the above specified locations

---

</details>

---

## Details Block Standards

<details>
<summary>Required Details Block Structure and Rules</summary>

---

### Mandatory Structure
- **Start with `---`** immediately after opening `<details>` tag
- **End with `---`** before closing `</details>` tag
- **Use descriptive summary text** - "Implementation Strategies," not "Details"

### Complete Details Block Template
```markdown
<details>
<summary>Clear description of what this section covers</summary>

---

- Content in bullet points
- All related information

#### Subsubsection (if needed)

Specific content for this subtopic

---

#### Another Subsubsection (if needed)

More specific content

---

</details>
```

### Summary Text Guidelines
- **Be descriptive and specific**: "Implementation Strategies," "Performance Metrics"
- **Use action-oriented language**: "Configure," "Implement," "Analyze"
- **Indicate content scope**: "Comprehensive Guide," "Essential Concepts"
- **Avoid vague text**: "Details," "More Information," "Click to Expand"

---

</details>

---

## Formatting Requirements

<details>
<summary>Essential Content and Technical Formatting Rules</summary>

---

### Content Format
- **All content must be bullet points** - break paragraphs into focused bullets
- **One concept per bullet point** - keep each bullet focused on single idea
- **No numbered elements** - use descriptive headings and bullet points only
- **Wrap symbols in backticks** - `<0.2%`, `$800K` to prevent markdown parsing issues

### Block Element Indentation (Critical)
- **All block elements must be indented 2 spaces from their parent**
- Applies to: code blocks (```), YAML blocks, Mermaid charts, tables
- Example:
  ```markdown
  - Parent bullet point
    ```yaml
    key: value
    ```
  - Parent with table
    | Col1 | Col2 |
    |------|------|
    | Data | Item |
  ```

### Technical Elements
- Use fenced code blocks with language specification
- Include proper indentation for all block elements
- Use tables for structured data with consistent formatting
- Bold important terms in first column of tables

---

</details>

---

## Quality Checklist

<details>
<summary>Essential Review Requirements</summary>

---

### Structure Review
- [ ] YAML front matter present with snake_case title
- [ ] Every subsection (###) has at most one details block
- [ ] Main sections (##) separated by `---` horizontal rules
- [ ] NO separators between ### level sections
- [ ] Details blocks start and end with `---` separators
- [ ] Subsubsections (####) within details blocks separated by `---`
- [ ] Summary text is descriptive and specific

### Content Review
- [ ] All content formatted as bullet points
- [ ] Block elements properly indented under parent items
- [ ] No numbered headings or bullet points
- [ ] All original content preserved when applying style
- [ ] Technical symbols wrapped in backticks

### Technical Review
- [ ] Code blocks have language specification
- [ ] Tables are consistently formatted
- [ ] All functions/APIs documented with parameters
- [ ] Examples are functional and relevant

---

</details>

--- 