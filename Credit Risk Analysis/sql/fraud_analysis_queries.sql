-- Credit Card Fraud Detection SQL Queries
-- Financial Analytics and Fraud Investigation

-- ============================================
-- 1. OVERALL FRAUD METRICS
-- ============================================

-- Overall fraud statistics
SELECT
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_transactions,
    COUNT(*) - SUM(is_fraud) AS legitimate_transactions,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_percent,
    SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount,
    SUM(amount) AS total_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) * 100.0 / SUM(amount), 2) AS fraud_amount_percent
FROM transactions;


-- ============================================
-- 2. FRAUD BY LOCATION
-- ============================================

SELECT
    location,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate,
    AVG(amount) AS avg_transaction_amount
FROM transactions
GROUP BY location
ORDER BY fraud_rate DESC;


-- ============================================
-- 3. FRAUD BY TRANSACTION TYPE
-- ============================================

SELECT
    transaction_type,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate,
    SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount
FROM transactions
GROUP BY transaction_type
ORDER BY fraud_rate DESC;


-- ============================================
-- 4. FRAUD BY MERCHANT CATEGORY
-- ============================================

SELECT
    merchant_category,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate,
    SUM(amount) AS total_amount
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_count DESC;


-- ============================================
-- 5. FRAUD BY TIME PATTERNS
-- ============================================

-- Fraud by hour of day
SELECT
    hour_of_day,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- Fraud by weekend vs weekday
SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
GROUP BY is_weekend;


-- ============================================
-- 6. HIGH-RISK TRANSACTION PATTERNS
-- ============================================

-- Large transactions (potential fraud)
SELECT
    transaction_id,
    customer_id,
    transaction_datetime,
    amount,
    merchant_category,
    location,
    is_fraud
FROM transactions
WHERE amount > 1000
ORDER BY amount DESC
LIMIT 50;


-- Foreign transactions
SELECT
    location,
    COUNT(*) AS num_transactions,
    SUM(is_fraud) AS fraud_count,
    AVG(amount) AS avg_amount,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
WHERE location = 'Foreign'
GROUP BY location;


-- Multiple failed PIN attempts
SELECT
    customer_id,
    COUNT(*) AS transactions_with_failed_pins,
    SUM(failed_pin_attempts) AS total_failed_attempts,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
WHERE failed_pin_attempts > 0
GROUP BY customer_id
HAVING SUM(failed_pin_attempts) >= 2
ORDER BY total_failed_attempts DESC;


-- ============================================
-- 7. CUSTOMER FRAUD ANALYSIS
-- ============================================

-- Customers with fraud incidents
SELECT
    customer_id,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    SUM(amount) AS total_spent,
    SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount,
    MAX(transaction_datetime) AS last_transaction
FROM transactions
WHERE customer_id IN (
    SELECT customer_id
    FROM transactions
    WHERE is_fraud = 1
)
GROUP BY customer_id
ORDER BY fraud_count DESC;


-- ============================================
-- 8. RAPID SUCCESSION TRANSACTIONS
-- ============================================

-- Transactions with very short time gaps (potential fraud)
SELECT
    transaction_id,
    customer_id,
    transaction_datetime,
    amount,
    time_since_last_txn,
    location,
    is_fraud
FROM transactions
WHERE time_since_last_txn < 5  -- Less than 5 minutes
ORDER BY customer_id, transaction_datetime;


-- ============================================
-- 9. CARD PRESENT VS NOT PRESENT
-- ============================================

SELECT
    CASE WHEN card_present = 1 THEN 'Card Present' ELSE 'Card Not Present' END AS card_status,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
GROUP BY card_present;


-- ============================================
-- 10. ONLINE VS IN-PERSON FRAUD
-- ============================================

SELECT
    CASE WHEN is_online = 1 THEN 'Online' ELSE 'In-Person' END AS channel,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate,
    AVG(amount) AS avg_transaction_amount
FROM transactions
GROUP BY is_online;


-- ============================================
-- 11. TRANSACTION AMOUNT ANALYSIS
-- ============================================

-- Fraud by transaction amount ranges
SELECT
    CASE
        WHEN amount < 50 THEN '$0-$50'
        WHEN amount < 100 THEN '$50-$100'
        WHEN amount < 200 THEN '$100-$200'
        WHEN amount < 500 THEN '$200-$500'
        WHEN amount < 1000 THEN '$500-$1000'
        ELSE '$1000+'
    END AS amount_range,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
GROUP BY amount_range
ORDER BY fraud_rate DESC;


-- ============================================
-- 12. CHIP TECHNOLOGY EFFECTIVENESS
-- ============================================

SELECT
    CASE WHEN chip_used = 1 THEN 'Chip Transaction' ELSE 'Non-Chip Transaction' END AS chip_status,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
GROUP BY chip_used;


-- ============================================
-- 13. DISTANCE FROM HOME ANALYSIS
-- ============================================

SELECT
    CASE
        WHEN distance_from_home = 0 THEN 'Home Location'
        WHEN distance_from_home < 50 THEN '0-50 miles'
        WHEN distance_from_home < 100 THEN '50-100 miles'
        WHEN distance_from_home < 500 THEN '100-500 miles'
        ELSE '500+ miles'
    END AS distance_range,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate
FROM transactions
GROUP BY distance_range
ORDER BY fraud_rate DESC;


-- ============================================
-- 14. MONTHLY FRAUD TRENDS
-- ============================================

SELECT
    strftime('%Y-%m', transaction_datetime) AS month,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate,
    SUM(amount) AS total_amount,
    SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END) AS fraud_amount
FROM transactions
GROUP BY month
ORDER BY month;


-- ============================================
-- 15. SUSPICIOUS PATTERN DETECTION
-- ============================================

-- Identify potentially suspicious patterns (not necessarily flagged as fraud)
SELECT
    transaction_id,
    customer_id,
    transaction_datetime,
    amount,
    merchant_category,
    location,
    failed_pin_attempts,
    distance_from_home,
    is_fraud
FROM transactions
WHERE
    (
        -- Large amount far from home
        (amount > 500 AND distance_from_home > 500)
        -- Multiple failed PINs
        OR failed_pin_attempts >= 2
        -- Late night large transaction
        OR (hour_of_day BETWEEN 0 AND 5 AND amount > 300)
        -- Very rapid transactions
        OR time_since_last_txn < 2
    )
    AND is_fraud = 0  -- Show legitimate transactions matching suspicious patterns
ORDER BY transaction_datetime DESC
LIMIT 100;
