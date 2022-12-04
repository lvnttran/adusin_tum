def payroll(name):
    num_hrs = float(input("Number of hrs. worked: "))
    print("Rate per hour: 80")
    overtime = float(input("Overtime(hrs.): "))
    gross_pay = (num_hrs * 80) + (overtime * 80 * 1.25)
    if gross_pay < 2250:
        SSS = float(gross_pay - 80)
        PhilHealth = float(gross_pay * 0.02)
        total_deduct = float(SSS + PhilHealth)
        net_pay = float(gross_pay - total_deduct)

    else:
        SSS = float(gross_pay - 800)
        PhilHealth = float(gross_pay * 0.02)
        total_deduct = float(SSS + PhilHealth)
        net_pay = float(gross_pay - 0.25 * total_deduct)

    print(name, "salary")
    print('Gross pay:', round(gross_pay, 2), "dollars")
    print('Net pay:', round(net_pay, 2), "dollars")


payroll("tam")

