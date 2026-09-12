import pandas as pd

df = pd.read_csv("train.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = [c.strip() for c in df.columns]

if 'nifty_before' in df.columns and 'nifty_after' in df.columns:
    df['points_change'] = df['nifty_after'] - df['nifty_before']

nifty_current = float(input("NIFTY current: "))
call_volume   = float(input("Call OI change: "))
put_volume    = float(input("Put OI change: "))
total_call    = float(input("Total Call OI: "))
total_put     = float(input("Total Put OI: "))

# ==========================================
# PRIMARY SIGNAL: SIMPLE OI RULE
# ==========================================
print(f"\n{'='*60}")
print(f" OPEN INTEREST (OI) SIGNAL")
print(f"{'='*60}")

# Simple rule:
# - If Call OI is positive → Buy PE
# - If Put OI is positive → Buy CE

oi_signals = []

if call_volume > 0:
    oi_signals.append("BUY PUT")
    print(f"\n📉 Call OI: +{call_volume:,} (POSITIVE)")
    print("   → Call writers are selling")
    print("   → Signal: BUY PE (PUT OPTION)")
elif call_volume < 0:
    oi_signals.append("BUY CALL")
    print(f"\n📈 Call OI: {call_volume:,} (NEGATIVE)")
    print("   → Call writers are covering (exiting)")
    print("   → Signal: BUY CE (CALL OPTION)")

if put_volume > 0:
    oi_signals.append("BUY CALL")
    print(f"\n📈 Put OI: +{put_volume:,} (POSITIVE)")
    print("   → Put writers are selling")
    print("   → Signal: BUY CE (CALL OPTION)")
elif put_volume < 0:
    oi_signals.append("BUY PUT")
    print(f"\n📉 Put OI: {put_volume:,} (NEGATIVE)")
    print("   → Put writers are covering (exiting)")
    print("   → Signal: BUY PE (PUT OPTION)")

# Determine final OI signal based on majority
if oi_signals.count("BUY CALL") >= oi_signals.count("BUY PUT"):
    oi_signal = "BUY CALL"
else:
    oi_signal = "BUY PUT"

print(f"\n{'='*60}")
print(f"🎯 PRIMARY OI SIGNAL: {oi_signal}")
print(f"{'='*60}")

# ==========================================
# KNN PATTERN MATCHING (Secondary Confirmation)
# ==========================================
feats = ['call_volume', 'put_volume', 'total_call', 'total_put']
live  = [call_volume, put_volume, total_call, total_put]

dist = 0
for f, v in zip(feats, live):
    s = df[f].std()
    if pd.isna(s) or s == 0: s = 1
    dist = dist + ((df[f] - v) / s) ** 2
df['distance'] = dist ** 0.5

near = df.nsmallest(3, 'distance')
call_votes = (near['WINNER'].str.strip() == 'BUY CALL').sum()
put_votes  = (near['WINNER'].str.strip() == 'BUY PUT').sum()

print(f"\n KNN HISTORICAL PATTERN:")
if call_votes >= put_votes:
    knn_signal = "BUY CALL"
    print(f"   BUY CALL (history {call_votes}-{put_votes})")
else:
    knn_signal = "BUY PUT"
    print(f"   BUY PUT (history {put_votes}-{call_votes})")

# ==========================================
# FINAL DECISION
# ==========================================
print(f"\n{'='*60}")
print("⚖️ FINAL TRADING DECISION")
print(f"{'='*60}")

if oi_signal == knn_signal:
    final_suggestion = oi_signal
    confidence = "HIGH"
    print(f"\n✅ STRONG SIGNAL: {final_suggestion}")
    print(f"   • OI Logic: {oi_signal}")
    print(f"   • Historical Pattern: {knn_signal}")
    print(f"   • Both AGREE → High Confidence Trade")
else:
    final_suggestion = oi_signal  # OI logic is primary
    confidence = "MEDIUM"
    print(f"\n⚠️ SIGNAL: {final_suggestion}")
    print(f"   • OI Logic (Primary): {oi_signal}")
    print(f"   • Historical Pattern: {knn_signal}")
    print(f"   • Following OI Logic (real-time data)")
    print(f"   • Confidence: MEDIUM (reduce quantity)")

suggestion = final_suggestion

# Expected Points
avg_points = 0
if 'points_change' in near.columns:
    avg_points = near['points_change'].mean()
    if avg_points >= 0: 
        print(f"\n📈 Expected NIFTY Increase: +{avg_points:.2f} points")
    else: 
        print(f"\n📉 Expected NIFTY Decrease: {avg_points:.2f} points")
    print(f"🎯 Target NIFTY Price: {nifty_current + avg_points:.2f}")

# ==========================================
# RISK MANAGEMENT
# ==========================================
print(f"\n{'='*60}")
print("️ RISK MANAGEMENT")
print(f"{'='*60}")

if 'points_change' in near.columns:
    wins = near[near['WINNER'].str.strip() == suggestion]['points_change']
    losses = near[near['WINNER'].str.strip() != suggestion]['points_change']
    
    if len(wins) > 0:
        avg_win = wins.mean()
        print(f"\nWhen {suggestion} worked:")
        print(f"   • Average Gain: +{avg_win:.2f} points")
    
    if len(losses) > 0:
        avg_loss = losses.mean()
        print(f"\nWhen {suggestion} FAILED:")
        print(f"   • Average Loss: {avg_loss:+.2f} points")
        print(f"   • STOP LOSS: {nifty_current + avg_loss:.2f}")

    # Trading Plan
    print(f"\n📋 TRADING PLAN:")
    print(f"   Entry: {nifty_current:.2f}")
    target1 = nifty_current + (abs(avg_points) * 0.5) if avg_points != 0 else nifty_current + 10
    target2 = nifty_current + abs(avg_points) if avg_points != 0 else nifty_current + 20
    sl = nifty_current - (abs(avg_points) * 0.5) if avg_points != 0 else nifty_current - 10
    print(f"   Target 1: {target1:.2f} (Book 50%)")
    print(f"   Target 2: {target2:.2f} (Exit All)")
    print(f"   Stop Loss: {sl:.2f}")

# ==========================================
# AUTO-SAVE
# ==========================================
print(f"\n{'='*60}")
save_data = input("💾 Save to train.csv? (y/n): ").strip().lower()

if save_data == 'y':
    new_row = {
        'nifty_before': nifty_current,
        'call_volume': call_volume,
        'put_volume': put_volume,
        'total_call': total_call,
        'total_put': total_put,
        'WINNER': 'PENDING',
        'points_change': 0,
        'nifty_after': 0
    }
    pd.DataFrame([new_row]).to_csv('train.csv', mode='a', header=False, index=False)
    print("✅ Saved! Update 'WINNER' after candle closes.")
else:
    print(" Not saved.")

print(f"\n{'='*60}")
print("END OF ANALYSIS")
print(f"{'='*60}")