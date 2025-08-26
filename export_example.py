#!/usr/bin/env python3
"""
Example script demonstrating how to use the export functionality
"""

from datetime import datetime
from utils import export_summary, check_export_dependencies

def main():
    # Check what export formats are available
    print("🔍 Checking export dependencies...")
    available_formats = check_export_dependencies()
    
    # Sample meeting summary data
    sample_summary = {
        "John Smith": "Completed the database migration project ahead of schedule. The new system is 30% faster and more reliable. Need to update documentation next week.",
        "Sarah Johnson": "Finished the UI redesign for the customer portal. User testing showed 25% improvement in user satisfaction. Will deploy to production on Friday.",
        "Mike Chen": "Fixed critical security vulnerabilities in the authentication system. Implemented two-factor authentication. All tests are passing.",
        "Emily Davis": "Coordinated with the marketing team on the product launch. Prepared training materials for customer support. Launch scheduled for next month."
    }
    
    meeting_date = datetime(2025, 8, 26)
    
    print(f"\n📝 Sample Meeting Summary:")
    for speaker, summary in sample_summary.items():
        print(f"👤 {speaker}: {summary[:50]}...")
    
    # Export to available formats
    print(f"\n🚀 Exporting to available formats: {available_formats}")
    
    # Create exports directory
    import os
    exports_dir = "exports"
    os.makedirs(exports_dir, exist_ok=True)
    
    # Export to all available formats
    results = export_summary(
        summary_dict=sample_summary,
        meeting_date=meeting_date,
        formats=available_formats,
        output_dir=exports_dir
    )
    
    print(f"\n✨ Export completed! Files saved in '{exports_dir}' directory:")
    for format_type, filename in results.items():
        if filename:
            print(f"  📄 {format_type.upper()}: {filename}")

if __name__ == "__main__":
    main()
