-- Credit Risk Analysis SQL Queries
-- Loan Application and Default Analysis

-- ============================================
-- 1. OVERALL DEFAULT METRICS
-- ============================================

-- Overall default rate and loan statistics
SELECT
    COUNT(*) AS total_applications,
    SUM(TARGET) AS defaults,
    COUNT(*) - SUM(TARGET) AS repaid,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct,
    AVG(AMT_CREDIT) AS avg_loan_amount,
    AVG(AMT_INCOME_TOTAL) AS avg_income,
    SUM(AMT_CREDIT) AS total_credit_issued
FROM applications;


-- ============================================
-- 2. DEFAULT BY DEMOGRAPHICS
-- ============================================

-- Default rate by gender
SELECT
    CODE_GENDER,
    COUNT(*) AS total_applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct,
    AVG(AMT_INCOME_TOTAL) AS avg_income
FROM applications
GROUP BY CODE_GENDER
ORDER BY default_rate_pct DESC;


-- Default rate by education level
SELECT
    NAME_EDUCATION_TYPE,
    COUNT(*) AS total_applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct,
    AVG(AMT_CREDIT) AS avg_loan_amount
FROM applications
GROUP BY NAME_EDUCATION_TYPE
ORDER BY default_rate_pct DESC;


-- Default rate by income type
SELECT
    NAME_INCOME_TYPE,
    COUNT(*) AS total_applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct,
    AVG(AMT_INCOME_TOTAL) AS avg_income
FROM applications
GROUP BY NAME_INCOME_TYPE
ORDER BY default_rate_pct DESC;


-- ============================================
-- 3. DEFAULT BY FINANCIAL FACTORS
-- ============================================

-- Default by income range
SELECT
    CASE
        WHEN AMT_INCOME_TOTAL < 100000 THEN 'Low (<100K)'
        WHEN AMT_INCOME_TOTAL < 200000 THEN 'Medium (100K-200K)'
        WHEN AMT_INCOME_TOTAL < 500000 THEN 'High (200K-500K)'
        ELSE 'Very High (500K+)'
    END AS income_range,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY income_range
ORDER BY default_rate_pct DESC;


-- Default by credit amount range
SELECT
    CASE
        WHEN AMT_CREDIT < 200000 THEN 'Small (<200K)'
        WHEN AMT_CREDIT < 500000 THEN 'Medium (200K-500K)'
        WHEN AMT_CREDIT < 1000000 THEN 'Large (500K-1M)'
        ELSE 'Very Large (1M+)'
    END AS credit_range,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY credit_range
ORDER BY default_rate_pct DESC;


-- ============================================
-- 4. AGE AND EMPLOYMENT ANALYSIS
-- ============================================

-- Default by age group
SELECT
    CASE
        WHEN -DAYS_BIRTH / 365 < 25 THEN '18-25'
        WHEN -DAYS_BIRTH / 365 < 35 THEN '25-35'
        WHEN -DAYS_BIRTH / 365 < 45 THEN '35-45'
        WHEN -DAYS_BIRTH / 365 < 55 THEN '45-55'
        ELSE '55+'
    END AS age_group,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY age_group
ORDER BY age_group;


-- Default by employment length
SELECT
    CASE
        WHEN DAYS_EMPLOYED >= 0 THEN 'Unemployed/Pension'
        WHEN -DAYS_EMPLOYED / 365 < 1 THEN '<1 year'
        WHEN -DAYS_EMPLOYED / 365 < 5 THEN '1-5 years'
        WHEN -DAYS_EMPLOYED / 365 < 10 THEN '5-10 years'
        ELSE '10+ years'
    END AS employment_length,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY employment_length
ORDER BY default_rate_pct DESC;


-- ============================================
-- 5. FAMILY AND HOUSING
-- ============================================

-- Default by family status
SELECT
    NAME_FAMILY_STATUS,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY NAME_FAMILY_STATUS
ORDER BY default_rate_pct DESC;


-- Default by number of children
SELECT
    CNT_CHILDREN,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
WHERE CNT_CHILDREN <= 5
GROUP BY CNT_CHILDREN
ORDER BY CNT_CHILDREN;


-- Default by housing type
SELECT
    NAME_HOUSING_TYPE,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY NAME_HOUSING_TYPE
ORDER BY default_rate_pct DESC;


-- ============================================
-- 6. ASSET OWNERSHIP
-- ============================================

-- Default by car and realty ownership
SELECT
    FLAG_OWN_CAR,
    FLAG_OWN_REALTY,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY FLAG_OWN_CAR, FLAG_OWN_REALTY
ORDER BY default_rate_pct DESC;


-- ============================================
-- 7. CONTRACT TYPE ANALYSIS
-- ============================================

-- Default by contract type
SELECT
    NAME_CONTRACT_TYPE,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct,
    AVG(AMT_CREDIT) AS avg_credit_amount,
    AVG(AMT_ANNUITY) AS avg_annuity
FROM applications
GROUP BY NAME_CONTRACT_TYPE;


-- ============================================
-- 8. HIGH-RISK PROFILES
-- ============================================

-- Identify high-risk applicant profiles
SELECT
    SK_ID_CURR,
    CODE_GENDER,
    -DAYS_BIRTH / 365 AS age,
    NAME_EDUCATION_TYPE,
    NAME_INCOME_TYPE,
    AMT_INCOME_TOTAL,
    AMT_CREDIT,
    CNT_CHILDREN,
    TARGET
FROM applications
WHERE
    AMT_CREDIT / AMT_INCOME_TOTAL > 5  -- High credit-to-income ratio
    OR CNT_CHILDREN > 3  -- Many dependents
    OR AMT_INCOME_TOTAL < 50000  -- Low income
ORDER BY AMT_CREDIT / AMT_INCOME_TOTAL DESC
LIMIT 100;


-- ============================================
-- 9. CREDIT-TO-INCOME RATIO ANALYSIS
-- ============================================

-- Default by credit-to-income ratio
SELECT
    CASE
        WHEN AMT_CREDIT / AMT_INCOME_TOTAL < 2 THEN 'Low (<2x)'
        WHEN AMT_CREDIT / AMT_INCOME_TOTAL < 4 THEN 'Medium (2-4x)'
        WHEN AMT_CREDIT / AMT_INCOME_TOTAL < 6 THEN 'High (4-6x)'
        ELSE 'Very High (6x+)'
    END AS credit_income_ratio,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
WHERE AMT_INCOME_TOTAL > 0
GROUP BY credit_income_ratio
ORDER BY default_rate_pct DESC;


-- ============================================
-- 10. EXTERNAL SOURCE SCORES
-- ============================================

-- Default rate by external source score ranges
SELECT
    CASE
        WHEN EXT_SOURCE_2 IS NULL THEN 'Missing'
        WHEN EXT_SOURCE_2 < 0.3 THEN 'Low (0-0.3)'
        WHEN EXT_SOURCE_2 < 0.5 THEN 'Medium (0.3-0.5)'
        WHEN EXT_SOURCE_2 < 0.7 THEN 'Good (0.5-0.7)'
        ELSE 'Excellent (0.7+)'
    END AS ext_score_range,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct
FROM applications
GROUP BY ext_score_range
ORDER BY default_rate_pct DESC;


-- ============================================
-- 11. OCCUPATION ANALYSIS
-- ============================================

-- Default rate by occupation (top 10)
SELECT
    OCCUPATION_TYPE,
    COUNT(*) AS applications,
    SUM(TARGET) AS defaults,
    ROUND(AVG(TARGET) * 100, 2) AS default_rate_pct,
    AVG(AMT_INCOME_TOTAL) AS avg_income
FROM applications
WHERE OCCUPATION_TYPE IS NOT NULL
GROUP BY OCCUPATION_TYPE
HAVING COUNT(*) > 1000  -- Only occupations with significant sample size
ORDER BY default_rate_pct DESC
LIMIT 10;


-- ============================================
-- 12. APPROVAL RECOMMENDATIONS
-- ============================================

-- Low-risk applicants (approve immediately)
SELECT
    COUNT(*) AS low_risk_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applications), 2) AS percentage
FROM applications
WHERE
    AMT_CREDIT / AMT_INCOME_TOTAL < 3
    AND (FLAG_OWN_CAR = 'Y' OR FLAG_OWN_REALTY = 'Y')
    AND -DAYS_EMPLOYED / 365 > 2
    AND AMT_INCOME_TOTAL > 100000;

-- High-risk applicants (review carefully or decline)
SELECT
    COUNT(*) AS high_risk_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applications), 2) AS percentage
FROM applications
WHERE
    AMT_CREDIT / AMT_INCOME_TOTAL > 6
    OR (CNT_CHILDREN > 3 AND AMT_INCOME_TOTAL < 150000)
    OR AMT_INCOME_TOTAL < 50000;
