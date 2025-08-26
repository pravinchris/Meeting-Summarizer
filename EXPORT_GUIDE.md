# Meeting Summarizer Export Integration Guide

## Overview
Your meeting summarizer now supports exporting summaries to 4 different formats:
- **Word (.docx)** - Professional document format
- **PDF (.pdf)** - Universal document format
- **Excel (.xlsx)** - Spreadsheet with formatting and word counts
- **CSV (.csv)** - Simple data format, always available

## Quick Start

### 1. Check Available Formats
```python
from utils import check_export_dependencies

# Check what export formats are available
available_formats = check_export_dependencies()
print(f"Available formats: {available_formats}")
```

### 2. Export to All Formats
```python
from datetime import datetime
from utils import export_summary

# Your speaker summaries (example)
summaries = {
    "John": "Completed database migration project...",
    "Sarah": "Finished UI redesign with 25% improvement...",
    "Mike": "Fixed security vulnerabilities..."
}

# Export to all available formats
results = export_summary(
    summary_dict=summaries,
    meeting_date=datetime.now(),
    formats=['word', 'pdf', 'excel', 'csv'],
    output_dir='meeting_exports'  # Optional: specify directory
)
```

### 3. Export to Specific Formats
```python
# Export only to PDF and Word
results = export_summary(
    summary_dict=summaries,
    meeting_date=datetime(2025, 8, 26),
    formats=['pdf', 'word']
)
```

## Integration with Your Existing Code

### Option 1: Add to your existing app.py
```python
# Add this after generating summaries
if st.button("📤 Export Summary"):
    results = export_summary(
        summary_dict=summaries,
        meeting_date=datetime.now(),
        formats=['word', 'pdf', 'excel', 'csv']
    )
    
    # Provide download links
    for format_type, filepath in results.items():
        if filepath:
            with open(filepath, 'rb') as f:
                st.download_button(
                    f"Download {format_type.upper()}",
                    f.read(),
                    file_name=os.path.basename(filepath)
                )
```

### Option 2: Use the enhanced app_with_export.py
```bash
# Run the enhanced version with built-in export functionality
streamlit run app_with_export.py
```

## File Formats Details

### Word (.docx)
- Professional formatting with headings
- Speaker names highlighted
- Date and generation timestamp
- Easy to edit and share

### PDF (.pdf)
- Universal format, works everywhere
- Professional styling with colors
- Cannot be easily edited (good for final reports)
- Consistent appearance across devices

### Excel (.xlsx)
- Structured data in spreadsheet format
- Includes word counts for each speaker
- Color-coded headers and speaker names
- Good for analysis and data processing

### CSV (.csv)
- Simple comma-separated format
- Works with any spreadsheet application
- Smallest file size
- Easy to import into databases

## Error Handling

The export functions are designed to be robust:

```python
# Check dependencies before exporting
available_formats = check_export_dependencies()

# Only export to available formats
results = export_summary(
    summary_dict=summaries,
    meeting_date=datetime.now(),
    formats=available_formats  # Uses only what's available
)

# Check results
successful = [fmt for fmt, path in results.items() if path]
failed = [fmt for fmt, path in results.items() if not path]

print(f"Successful: {successful}")
print(f"Failed: {failed}")
```

## Dependencies

Required packages for full export functionality:
```bash
pip install python-docx reportlab openpyxl pandas
```

- `python-docx`: Word document export
- `reportlab`: PDF export  
- `openpyxl`: Excel export
- `pandas`: Always available for CSV

## File Naming Convention

Files are automatically named with timestamps:
- `meeting_summary_YYYYMMDD_HHMMSS.docx`
- `meeting_summary_YYYYMMDD_HHMMSS.pdf`
- `meeting_summary_YYYYMMDD_HHMMSS.xlsx`
- `meeting_summary_YYYYMMDD_HHMMSS.csv`

You can also specify custom filenames:
```python
export_to_word(summaries, meeting_date, "my_meeting.docx")
```

## Example Usage in Your Workflow

```python
# Your existing transcription and summarization code
transcript = transcribe_audio(wav_path, engine="faster-whisper")
speakers = extract_with_llm(transcript, "llama3.2:1b")
summaries = summarize_updates(speakers, "llama3.2:1b")

# NEW: Export the results
meeting_date = datetime.now()
export_results = export_summary(
    summary_dict=summaries,
    meeting_date=meeting_date,
    formats=['word', 'pdf', 'csv'],
    output_dir='meeting_outputs'
)

print("Export completed!")
for format_type, filepath in export_results.items():
    if filepath:
        print(f"✅ {format_type.upper()}: {filepath}")
```

This integration maintains compatibility with your existing code while adding powerful export capabilities!
