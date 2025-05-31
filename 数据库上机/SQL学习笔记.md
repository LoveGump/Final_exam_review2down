# SQL

## 2.1SQL通用语法

1. sql可以单行或者多行书写，以分号结尾
2. sql空格/缩进 增强可读性
3. sql不区分大小写，关键字建议大写
   - 单行注释： -- 注释 或者 # （mysql）
   - 多行注释：/**/

## 2.2SQL分类

DDL（data defination language）

DML（data manipulation language）

DQL（data query language）

DCL（data control language）

![image-2025050122422663](/Users/gump/大二资料（更新版）/database/黑马笔记/insert/image-20250501224226634.png)

## 2.3DDL

### 2.3.1数据库操作

- 查询所有的数据库

```sql
SHOW DATABASES;
```

- 查询当前数据库

```sql
SELECT DATABASES();
```

- 创建

```sql
CREATE DATABASES [IF NOT EXISTS]数据库名 [DEFAULT CHARSET 字符集][COLLATE 排序规则]；
```

- 删除

```SQL
DROP DATABASES [IF EXISTS] 数据库名；
```

- 使用

```
USE 数据库名；
```

### 2.3.2表操作

**2.3.2.1** **表操作查询创建**

- 查询当前表中的所有表

```sql
SHOW TABLES
```

- 查询表结构

```sql
DESC 表名；
```

- 查询指定表的建表语句

```SQL
 SHOW CREATE TABLE 表名；
```



- 

```sql
create table emp(
	id int comment '编号',
  workno varchar(10) comment '员工工号',
  workname varchar(10) comment '员工姓名',
  gender char(1) comment '性别',
  age tinyint unsigned comment '年龄',
  idcard char(18) comment '身份证号',
  entrydate date comment '入职时间'
  
)comment '员工表';
```



- 