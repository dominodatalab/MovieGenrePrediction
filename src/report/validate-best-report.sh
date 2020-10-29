jupyter nbconvert --to html 'notebooks/validate-best.ipynb' --TemplateExporter.exclude_input=True --execute
mv -f notebooks/validate-best.html src/report/generated/validate_best_report.html