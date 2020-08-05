import csv, collections, time, sys, pickle

def def_user(row):
    if row['device_id'] == 'a99f214a':
        user = 'ip-' + row['device_ip'] + '-' + row['device_model']
    else:
        user = 'id-' + row['device_id']
    return user

def is_app(row):
    return True if row['site_id'] == '85f751fd' else False

def has_id_info(row):
    return False if row['device_id'] == 'a99f214a' else True


id_cnt = collections.defaultdict(int)
ip_cnt = collections.defaultdict(int)
user_cnt = collections.defaultdict(int)
user_hour_cnt = collections.defaultdict(int)

start = time.time()
maxline = 1000000 

def scan(path):
	for i, row in enumerate(csv.DictReader(open(path)), start=1):
		if i == maxline:
			break
		if i % 1000000 == 0:
			sys.stderr.write('scan {0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))

		user = def_user(row)
		id_cnt[row['device_id']] += 1
		ip_cnt[row['device_ip']] += 1
		user_cnt[user] += 1
		user_hour_cnt[user+'-'+row['hour']] += 1
	
	print('======== scan complete ========')

def process(src_path, dst_path, is_train):

	processed_data = []
	feature_dict = {}	
	fields = ['pub_id', 'pub_domain', 'pub_category', 'banner_pos', 'device_model', 'device_conn_type', 'C14', 'C17', 'C20', 'C21']

	# 0 for unknown feature
	feature_index = 1

	for i, row in enumerate(csv.DictReader(open(src_path)), start=1):
		if i == maxline:
			break
		if i % 1000000 == 0:
			sys.stderr.write('{0:6.0f}    {1}m\n'.format(time.time()-start,int(i/1000000)))

		features = []	
		feature_ids = []
		click = 0

		device_id_count = id_cnt[row['device_id']]	
		device_ip_count = ip_cnt[row['device_ip']]

		user, hour = def_user(row), row['hour']
		user_count = user_cnt[user]
		user_hour_count = user_hour_cnt[user+'-'+hour]

		if is_app(row):
			row['pub_id'] = row['app_id']
			row['pub_domain'] = row['app_domain']
			row['pub_category'] = row['app_category']
		else:
			row['pub_id'] = row['site_id']
			row['pub_domain'] = row['site_domain']
			row['pub_category'] = row['site_category']
		
		for field in fields:
			features.append(field + '-' + row[field])

		if device_ip_count > 1000:
			features.append('device_ip-' + row['device_ip'])
		else: 
			features.append('device_ip-less-' + str(device_ip_count))
		
		if device_id_count > 1000:
			features.append('device_id-' + str(row['device_id'])) 
		else: 
			features.append('device_id-less-' + str(device_id_count))

		if user_hour_count > 30:
			features.append('user_hour_count-0')
		else: 
			features.append('user_hour_count-' + str(user_hour_count))

		feature_indices = []
		for feature in features:
			if feature not in feature_dict:
				feature_dict[feature] = feature_index
				feature_index += 1
			feature_indices.append(feature_dict[feature])
		
		processed_data.append({'feature_indices': feature_indices, 'click': click})
	
	print('======== process complete ========')
	print('Total feature count: ', feature_index) 
	f_data = open(dst_path + '_data', 'wb')
	f_dict = open(dst_path + '_feature_dict', 'wb')
	pickle.dump(processed_data, f_data)
	pickle.dump(feature_dict, f_dict)


scan('../raw_data/train')
process('../raw_data/train', '../data/train', True)