from urllib.request import urlopen
import zipfile


def download_file(url_str, path):
    url = urlopen(url_str)
    output = open(path, 'wb')       
    output.write(url.read())
    output.close()  
    
def extract_file(archive_path, target_dir):
    zip_file = zipfile.ZipFile(archive_path, 'r')
    zip_file.extractall(target_dir)
    zip_file.close()



def input_Wset_Lset(df_atp):
    df_atp['Wsets'].fillna(0,inplace=True)
    df_atp['Lsets'].fillna(0,inplace=True)



def previous_w_percentage(player,date, df_atp):
    minimum_played_games = 2
    df_previous  = df_atp[df_atp["Date"] < date]
    previous_wins = df_previous[df_previous["Winner"] == player].shape[0]
    previous_losses = df_previous[df_previous["Loser"] == player].shape[0]
    
    if  minimum_played_games > (previous_wins + previous_losses):
        return 0
    return previous_wins / (previous_wins + previous_losses)   


def positions(df_atp):
    df_atp["Winner_position"] = df_atp.apply(lambda row: 1 if row["Winner"] > row["Loser"] else 0, axis=1)
    df_atp[["Winner", "Loser", "Winner_position"]].head(5)