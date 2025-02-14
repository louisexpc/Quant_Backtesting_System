for i in range(self.ds.length-1):
        #     self.next()
        #     #print(f"day {i}:\n{self.spot.current_data}")
        #     #print(f"balance:\n{self.account.spot_balance}")
        #     #print(f"order_book:\n{self.spot.order_book}")
        #     #print(f"position:\n{self.spot.positions}")
        #     self.ds.run()
        # """Final 結算"""
        # self.spot.calculate_unrealized_pnl()
        # print(self.spot.positions)
        # print(self.account.spot_balance)
        # print(f"盈虧:{self.spot.history_positions['pnl'].sum()}")
        # self.spot.history_positions.to_csv("history.csv")