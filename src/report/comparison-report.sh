jupyter nbconvert --to html 'notebooks/comparison.ipynb' --TemplateExporter.exclude_input=True --execute
mv -f notebooks/comparison.html src/report/generated/comparison_report.html