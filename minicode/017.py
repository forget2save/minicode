# coding:utf-8
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from security import username, password, driverPath


def click(driver, xpath):
    driver.find_element_by_xpath(xpath).click()


def main():
    website = "https://healthreport.zju.edu.cn/ncov/wap/default/index"
    driver = webdriver.Edge(executable_path=driverPath)
    driver.get(website)
    if driver.title == "统一身份认证平台":
        driver.find_element_by_name("username").send_keys(username)
        driver.find_element_by_name("password").send_keys(password, Keys.ENTER)
    assert driver.title == "每日上报"
    driver.maximize_window()
    choiceMap = {
        "sffrqjwdg": 2,
        "sfqtyyqjwdg": 2,
        "tw": 2,
        "sfcxtz": 2,
        "sfjcbh": 2,
        "sfjcqz": 2,
        "sfyqjzgc": 2,
        "sfcyglq": 2,
        "jrsfqzys": 2,
        "jrsfqzfy": 2,
        "sfhsjc": 1,
        "sfcxzysx": 2,
        "sfsqhzjkk": 1,
        "sqhzjkkys": 1,
        "zgfx14rfh": 2,
        "sfzx": 2,
        "sfzgn": 1,
        "sfymqjczrj": 2,
        "sfqrxxss": 1,
    }
    for key, value in choiceMap.items():
        click(driver, f'//*[@name="{key}"]/div/div[{value}]/span')
    click(driver, "//div[@name='area']/input")
    sleep(1)
    click(driver, "//a[@class='wapcf-btn-qx']")
    sleep(1)
    driver.close()


if __name__ == "__main__":
    main()
