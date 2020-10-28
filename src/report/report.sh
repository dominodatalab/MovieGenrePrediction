jupyter nbconvert --to html 'notebooks/Report.ipynb' --TemplateExporter.exclude_input=True --execute
mv -f notebooks/Report.html src/report/generated/email.html