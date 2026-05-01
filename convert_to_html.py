import markdown2

# Read the markdown file
with open('reports/Project_Report.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert to HTML
html_content = markdown2.markdown(md_content)

# HTML template
html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Email Spam Classification Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; text-align: center; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        h3 { color: #7f8c8d; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        code { background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
        pre { background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
""" + html_content + """
</body>
</html>"""

# Save HTML
with open('reports/Project_Report.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print('HTML created: reports/Project_Report.html')