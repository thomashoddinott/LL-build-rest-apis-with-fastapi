CREATE TABLE trades (
	time TIMESTAMP,
	user VARCHAR(32),
	symbol VARCHAR(32),
	price INTEGER, -- In ¢
	volume INTEGER,
	side VARCHAR(4) -- "buy" or "sell"
);