def buy(deposits, splitted):
    user_id_corrected = int(splitted[0]) - 1
    amount = int(splitted[1])
    price = int(splitted[2])
    deposits[user_id_corrected] -= (amount * price)
    return deposits[user_id_corrected] 
    
def sell(deposits, splitted):
    amount = int(splitted[1])
    price = int(splitted[2])
    user_id_corrected = int(splitted[0]) - 1
    deposits[user_id_corrected] += (amount * price)
    return deposits[user_id_corrected] 
    
def deposit(deposits, splitted):
    user_id_corrected = int(splitted[0]) - 1
    amount = int(splitted[1])
    deposits[user_id_corrected] += int(amount)
    return deposits[user_id_corrected]


def cryptoTrading(deposits, operations):
    moneyLog = []
    
    for operation in operations:
        # obtain operation op
        splitted = operation.split(" ")
        
        if splitted[0] == "buy":
            moneyLog.append(buy(deposits, splitted))
            
        elif splitted[0] == "sell":
            moneyLog.append(sell(deposits, splitted))
            
        elif splitted[0] == "deposit":
            moneyLog.append(deposit(deposits, splitted))
            
    
    
    return moneyLog
