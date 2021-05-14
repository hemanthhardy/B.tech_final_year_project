from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# DANGER! This is insecure. See http://twil.io/secure
account_sid = 'AC8bf88e286587e81a40610f4098d8d1af'
auth_token = '3d72840a71e6592d20e1bc7f9199704a'
client = Client(account_sid, auth_token)
call = client.calls.create(
                        twiml='<Response><Say></Say></Response>',
                        to='+918610463655',
                        from_='+16413292467'
                    )

print(call.sid)

