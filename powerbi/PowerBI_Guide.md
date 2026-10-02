# Power BI Dashboard Guide

## Import
Load `outputs/powerbi_employee_attrition.csv`.

## KPI cards
Create:
- Total Employees = DISTINCTCOUNT(EmployeeNumber) if EmployeeNumber exists
- Employees Left = CALCULATE(COUNTROWS(data), data[Attrition] = 1)
- Attrition Rate = DIVIDE([Employees Left], [Total Employees])
- Average Monthly Income = AVERAGE(data[MonthlyIncome])
- Average Years at Company = AVERAGE(data[YearsAtCompany])

## Recommended visuals
1. Attrition by Department – clustered column
2. Attrition by Job Role – bar
3. Attrition by OverTime – column
4. Attrition by Salary Band – column
5. Attrition by Age Band – column
6. Attrition by Years at Company – line/column
7. Predicted Attrition Probability – histogram or column
8. Actual vs Predicted Attrition – matrix
9. SHAP feature importance image as a report element

## Slicers
Department, JobRole, Gender, OverTime, AgeBand, IncomeBand.

## Design
Use one page with:
- top KPI cards,
- middle diagnostic charts,
- bottom risk/prediction section,
- slicers on the left.
Avoid clutter and keep labels readable.
