import psycopg2  # or pymongo for MongoDB

def update_database(result):
    conn = psycopg2.connect(dbname='tollproject', user='postgres', password='postgres', host='localhost')
    cur = conn.cursor()
    print("connection successfull")
    sql = "INSERT INTO numberplate (plateno) VALUES (%s)"
    values = (result,)
    cur.execute(sql, values)
    conn.commit()
    cur.close()
    conn.close()

def clear_database():
    conn = psycopg2.connect(dbname='tollproject', user='postgres', password='postgres', host='localhost')
    cur = conn.cursor()
    print("connection successfull")
    sql = "delete from numberplate"
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
