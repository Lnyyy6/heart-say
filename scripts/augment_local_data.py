"""本地数据扩充：20 个场景 × 25 组家长/青少年表达，生成 1k 条多轮对话与 500 条偏好对。

不调用外部 API；assistant 回复由离线共情编排器、固定危机协议与边界话术生成，
再用 build_training_data.py 统一做 PII、去重与安全过滤。它不是“强模型合成”的
替代品，data card 会如实标注。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_say.agent.emotion import detect_emotion  # noqa: E402
from heart_say.llm.client import GenerationContext, OfflineComposer, boundary_reply  # noqa: E402
from heart_say.rag import get_retriever  # noqa: E402
from heart_say.safety import assess_risk, crisis_response, detect_audience, redact_pii  # noqa: E402

CHILD = ["儿子", "女儿", "孩子"]
GRADES = ["初一", "初二", "初三", "高一", "高二", "高三"]
AGES = [12, 13, 14, 15, 16, 17]

SCENARIOS = [
    {
        "name": "考试失利与排名焦虑",
        "parent_behavior": ["这次月考掉到年级两百多名，回家就把门关上", "期中数学没及格，说不想去学校了", "模拟考排名下滑，她把自己关在房间里哭", "平时成绩不错，一到大考就发挥失常", "最近作业写到很晚，成绩反而掉了"],
        "parent_worry": ["我不知道该怎么安慰她", "我一开口他就嫌烦", "我比他还焦虑，晚上睡不着", "我怕说重了伤她自尊", "我想帮他但不知道从哪下手"],
        "teen_situation": ["这次月考掉到年级两百多名", "数学又没及格，老师点名批评我", "模拟考排名一直往下掉", "一到大考我就脑子空白", "我明明复习了，成绩还是上不去"],
        "teen_feeling": ["我妈肯定又要说我", "我觉得自己特别没用", "我不敢回家面对他们", "我越努力越觉得自己笨", "我不想再提成绩了"],
        "parent_followups": ["我说了她也不听，我该怎么办？", "我是不是管得太多了？", "我一看到成绩单就难受。"],
        "teen_followups": ["我妈一开口我就想顶回去。", "我是不是真的不适合读书？", "我现在一想到考试就心慌。"],
    },
    {
        "name": "手机与游戏使用冲突",
        "parent_behavior": ["一回家就抱着手机，说他也不听", "半夜还在打游戏，第二天起不来", "说好只玩半小时，结果玩到凌晨", "收了手机就跟我吵，摔门", "写作业时偷偷刷短视频"],
        "parent_worry": ["我怕他沉迷下去", "我是不是管得太严了", "我不知道怎么定规矩", "我一管他就说我控制他", "我怕越管越僵"],
        "teen_situation": ["我妈一回家就收我手机", "我打完这一局就停不下来", "我只想放松一下，她就开始骂", "同学都在玩，我不玩就插不上话", "我一拿手机我妈就盯着我"],
        "teen_feeling": ["我觉得她根本不信任我", "我越被管越想对着干", "我也不知道怎么停下来", "我只是想有一点自己的时间", "我烦透了她天天念"],
        "parent_followups": ["我定了规矩他也不执行。", "我是不是该直接把手机没收？", "我一说他他就嫌我烦。"],
        "teen_followups": ["她一收手机我就什么都不想说了。", "我是不是只能偷偷玩？", "我怎么跟她说她才不会炸？"],
    },
    {
        "name": "孩子拒绝沟通、冷战",
        "parent_behavior": ["回家就关门，我问一句她答一句", "最近一句话都不愿意跟我说", "我一开口他就戴上耳机", "吃饭也不上桌，躲在房间里", "问他学校的事就说“没事”"],
        "parent_worry": ["我不知道他在想什么", "我是不是哪里做错了", "我怕他一直这样下去", "我想靠近他又怕被推开", "我觉得自己像个外人"],
        "teen_situation": ["我妈天天追着问我学校的事", "我一开口她就讲道理", "我不想说话，说了也没用", "我一回家就想躲进房间", "他们根本不听我讲完"],
        "teen_feeling": ["我觉得说了也会被否定", "我只想安静一会儿", "我懒得再解释", "我一说话就吵起来", "我不觉得他们真的懂我"],
        "parent_followups": ["我越问越僵，怎么办？", "我是不是应该先不问了？", "我怎么才能让他愿意开口？"],
        "teen_followups": ["她一讲道理我就不想说了。", "我是不是太冷漠了？", "我怎么才能有自己的空间？"],
    },
    {
        "name": "校园同伴排挤或霸凌",
        "parent_behavior": ["孩子说班里几个人总给他起外号", "她最近不敢去食堂，说有人堵她", "同学在群里发他的丑照，还艾特所有人", "孩子被要钱，不给就威胁他", "她回来说被同桌孤立了"],
        "parent_worry": ["我怕他出事", "我不知道该不该找老师", "我怕闹大了孩子更遭殃", "我咽不下这口气", "我不知道怎么保护她"],
        "teen_situation": ["班里几个人一直给我起外号", "他们堵在厕所跟我要钱", "同学在群里发我的丑照", "我被同桌孤立了，没人跟我说话", "他们说我要是告诉老师就弄我"],
        "teen_feeling": ["我假装不在意，其实很难受", "我不敢去学校", "我觉得是不是我的问题", "我怕告诉老师会更糟", "我每天都提心吊胆"],
        "parent_followups": ["我要不要直接去学校找老师？", "孩子不让我告诉老师，我该怎么办？", "我怎么确认他现在安不安全？"],
        "teen_followups": ["我不敢跟老师说，怕他们报复。", "我是不是太懦弱了？", "我明天不想去学校。"],
    },
    {
        "name": "早恋与边界",
        "parent_behavior": ["女儿高二，最近跟一个男生走得很近", "我偷看了她手机，发现聊天很暧昧", "儿子天天跟一个女生聊到半夜", "她周末总找借口出去见同学", "我发现他给同学写了情书"],
        "parent_worry": ["我怕影响学习", "我不知道该不该挑明", "我怕她被骗", "我一说她就炸", "我怕管多了把她推远"],
        "teen_situation": ["我妈偷看了我手机", "我喜欢一个人，但不敢让家里知道", "我妈说早恋就是不自爱", "我只是想跟喜欢的人聊天", "他们一发现就禁止我出门"],
        "teen_feeling": ["我觉得她根本不尊重我", "我很怕被他们发现", "我不知道这算不算喜欢", "我又想靠近又怕受伤", "我觉得自己做什么都错"],
        "parent_followups": ["我该不该直接禁止他们联系？", "我一提这事她就关门。", "我怎么跟她说边界才不伤感情？"],
        "teen_followups": ["我妈越禁止我越想见她。", "我是不是不该喜欢别人？", "我怎么保护自己又不被控制？"],
    },
    {
        "name": "离婚或重组家庭中的亲子关系",
        "parent_behavior": ["我和孩子他爸上个月离婚了，孩子跟着我", "我再婚两年，继父想管女儿学习，她很排斥", "孩子周末去爸爸那边，回来就闷闷不乐", "她总说“你不是我爸”", "孩子在两个家里来回，情绪很不稳定"],
        "parent_worry": ["我怕孩子觉得被抛弃", "我夹在中间很难受", "我是不是该让女儿懂事点", "我怕再婚伤害了她", "我不知道怎么跟她解释"],
        "teen_situation": ["我爸妈上个月离婚了，我爸搬走了", "我妈再婚了，继父总想管我", "我周末去爸爸家，回来我妈就冷着脸", "他们吵架时总让我选边", "我觉得自己很多余"],
        "teen_feeling": ["我不知道该安慰谁", "我不想叫陌生人爸爸", "我觉得没人真正在乎我", "我两边都不属于", "我不想再听他们说为了我"],
        "parent_followups": ["我是不是该逼孩子接受继父？", "孩子说我不是她亲爸，我很难受。", "我怎么让她觉得这个家还是她的？"],
        "teen_followups": ["我不想回那个家。", "我是不是不该怪我妈？", "我怎么跟他们说我需要什么？"],
    },
    {
        "name": "二胎与公平感",
        "parent_behavior": ["老二出生后，老大突然开始尿床、要抱", "老大偷偷打妹妹，说我偏心", "两个孩子一吵架就让我评理", "老大总说“你只爱弟弟”", "我精力不够，老大变得越来越黏人"],
        "parent_worry": ["我是不是真的偏心", "我两边都顾不过来", "我怕老大心里留下阴影", "我一吼完就后悔", "我不知道怎么分时间"],
        "teen_situation": ["我妈什么都让我让着弟弟", "弟弟一哭就是我的错", "我爸妈只陪妹妹不陪我", "我做什么都要先考虑弟弟", "我觉得自己像多余的"],
        "teen_feeling": ["我很讨厌我弟弟", "我觉得他们根本不爱我", "我越想越委屈", "我不想再懂事了", "我只想被看见一次"],
        "parent_followups": ["我该怎么让老大觉得没有被忽略？", "两个孩子一吵我就崩溃。", "我是不是该单独陪老大？"],
        "teen_followups": ["我一让再让，他们还是觉得不够。", "我是不是不该讨厌弟弟？", "我怎么跟我妈说我需要她？"],
    },
    {
        "name": "睡眠不足与熬夜",
        "parent_behavior": ["孩子每天凌晨两三点才睡，早上起不来", "她老说睡不着，脑子停不下来", "他熬夜刷手机，白天上课打瞌睡", "孩子最近总说头疼、没精神", "周末补觉到中午，作息全乱了"],
        "parent_worry": ["我怕影响身体和学习", "我不知道怎么帮他调整", "我一催他睡他更烦", "我是不是该收手机", "我怕他是不是抑郁了"],
        "teen_situation": ["我每天凌晨两三点才睡得着", "我一躺下脑子就停不下来", "我早上起不来，上课一直走神", "我越困越睡不着", "我周末补觉也缓不过来"],
        "teen_feeling": ["我觉得自己快撑不住了", "我怕我妈又说我不自律", "我白天像行尸走肉", "我也不知道怎么停下来", "我只想好好睡一觉"],
        "parent_followups": ["我催他睡觉他就发火。", "我是不是该带他去看医生？", "我怎么帮他调整作息？"],
        "teen_followups": ["我一闭眼就开始想乱七八糟的事。", "我是不是身体出问题了？", "我怎么跟我妈说我不是故意的？"],
    },
    {
        "name": "身材焦虑与进食",
        "parent_behavior": ["女儿最近总说自己胖，吃得很少", "她吃完饭就躲进厕所", "孩子体重掉得很快，还坚持跑步", "她只吃水煮菜，拒绝吃肉", "她总拿自己和网上的女生比"],
        "parent_worry": ["我怕她得厌食症", "我不知道该不该逼她吃", "我一说她就说我嫌她胖", "我怕她身体垮掉", "我不知道怎么跟她谈身体"],
        "teen_situation": ["我158，90斤还觉得自己胖", "我吃完就想吐", "我最近只敢吃水煮菜", "我妈总说我瘦了不好看", "我每天都要称重好几次"],
        "teen_feeling": ["我看到镜子就难受", "我觉得只有瘦才值得被喜欢", "我怕一吃就失控", "我不想让别人看见我吃饭", "我觉得自己永远不够好"],
        "parent_followups": ["我一劝她吃她就哭。", "我是不是该带她去医院？", "我怎么跟她谈身体才不伤人？"],
        "teen_followups": ["我一吃就特别愧疚。", "我是不是已经不正常了？", "我怎么才能不讨厌自己的身体？"],
    },
    {
        "name": "厌学与拒学",
        "parent_behavior": ["孩子这学期开始天天说肚子疼不去上学", "他一到校门口就掉头回家", "她把自己关在房间，说死也不去学校", "孩子成绩一路下滑，现在干脆不写作业", "他每天早上都跟我磨一个多小时"],
        "parent_worry": ["我怕他彻底放弃", "我打也打了骂也骂了", "我不知道是不是学校出了事", "我怕逼他更糟", "我上班也要被他拖住"],
        "teen_situation": ["我就是不想上学", "我一进教室就喘不过气", "我觉得学校没有一处让我舒服", "我去了也听不懂，还不如不去", "我每天早上都希望自己生病"],
        "teen_feeling": ["我讨厌别人跟我讲大道理", "我觉得自己格格不入", "我一想到学校就害怕", "我不想再假装正常", "我只想在家待着"],
        "parent_followups": ["我逼他去他就跟我拼命。", "我是不是该先请假？", "我怎么知道学校到底发生了什么？"],
        "teen_followups": ["你们只会说未来、责任，我听够了。", "我是不是废了？", "我怎么才能不被逼着去？"],
    },
    {
        "name": "注意力与多动困扰",
        "parent_behavior": ["老师说他上课坐不住，总走神", "作业写两分钟就起来晃", "他丢三落四，书包永远一团糟", "我一辅导作业就控制不住发火", "他做事总是三分钟热度"],
        "parent_worry": ["我怀疑他是不是多动症", "我不知道该不该去医院", "我怕给他贴标签", "我天天吼他也没用", "我怕影响他自信"],
        "teen_situation": ["我上课总是走神，老师点我名", "我作业写到一半就想干别的", "我老是丢东西，被同学笑", "我妈一辅导作业就吼我", "我很难把一件事做完"],
        "teen_feeling": ["我觉得自己脑子有问题", "我不是不想认真，是控制不住", "我越被骂越不想学", "我觉得自己很没用", "我不想再被当成懒"],
        "parent_followups": ["我该不该带他去做评估？", "我一辅导作业就爆炸。", "我怎么帮他而不伤自尊？"],
        "teen_followups": ["我是不是多动症？", "我一坐下就难受。", "我怎么跟我妈说我不是故意的？"],
    },
    {
        "name": "网络比较与自我价值",
        "parent_behavior": ["孩子天天刷短视频，说自己一无是处", "她总跟网上的女生比长相和成绩", "他看到别人旅游、拿奖就情绪低落", "孩子说同学都比他过得好", "他越来越不愿意出门见人"],
        "parent_worry": ["我怕他被网络带偏", "我不知道怎么建立自信", "我一说他就说我落伍", "我怕他越来越封闭", "我不知道他真正想要什么"],
        "teen_situation": ["我刷到的每个人都比我好看、比我优秀", "我同学都在晒成绩和旅行", "我越刷越觉得自己失败", "我不敢发朋友圈，怕没人点赞", "我总觉得别人活得比我好"],
        "teen_feeling": ["我觉得自己什么都不行", "我很怕被比较", "我越看越焦虑", "我不知道自己有什么价值", "我不想让别人看见真实的我"],
        "parent_followups": ["我该怎么帮他建立自信？", "我一说他他就嫌我烦。", "我是不是该限制他刷手机？"],
        "teen_followups": ["我一放下手机就觉得自己更差。", "我是不是很虚荣？", "我怎么才能不在乎别人？"],
    },
    {
        "name": "亲子冲突后的修复",
        "parent_behavior": ["昨天我吼了孩子，今天他一直不理我", "我一时冲动说了“你真没用”，现在很后悔", "我们大吵一架，她摔门走了", "我动手打了他一下，他没哭，我更难受", "冷战三天了，我不知道怎么开口"],
        "parent_worry": ["我怕他记恨我", "我不知道怎么道歉", "我是不是个坏家长", "我怕关系回不去了", "我一开口又吵起来"],
        "teen_situation": ["我妈昨天吼我，说我什么都不行", "我爸打了我，我不想跟他说话", "我们吵完谁也不理谁", "她骂完我又当没事发生", "我现在一看到他们就烦"],
        "teen_feeling": ["我不想原谅她", "我觉得很丢脸", "我也不知道怎么面对他们", "我很委屈但说不出口", "我怕一开口又吵"],
        "parent_followups": ["我该怎么跟孩子道歉？", "我一开口又变成说教。", "我是不是该先不逼他说话？"],
        "teen_followups": ["她一道歉我就想哭。", "我是不是也有错？", "我怎么才能把这件事过去？"],
    },
    {
        "name": "家长自身情绪耗竭",
        "parent_behavior": ["我下班回家还得做饭辅导作业，孩子还嫌我烦", "我每天像陀螺一样转，没人搭把手", "我最近总想哭，觉得自己撑不住", "我一看到孩子就烦，之后又内疚", "我连好好吃顿饭的时间都没有"],
        "parent_worry": ["我是不是个失败的母亲", "我有时候真不想管了", "我怕把情绪撒在孩子身上", "我不知道跟谁说我累", "我觉得自己快耗干了"],
        "teen_situation": ["我妈每天都很累，还总对我发火", "我一回家她就叹气", "她说都是为了我才这么辛苦", "我不知道怎么让她轻松一点", "我一看到她疲惫就内疚"],
        "teen_feeling": ["我觉得是我拖累了她", "我不敢再提自己的需要", "我一听到“为了你”就窒息", "我想帮她但不知道怎么做", "我也很累但没人问"],
        "parent_followups": ["我该怎么开口向伴侣求助？", "我一累就控制不住吼孩子。", "我是不是太自私了？"],
        "teen_followups": ["我一看到她累就不敢说话。", "我是不是不该有自己的需要？", "我怎么帮她又不委屈自己？"],
    },
    {
        "name": "被老师批评后的羞耻",
        "parent_behavior": ["孩子被老师当众批评，回来一直不说话", "老师打电话说他在课堂上捣乱", "她被罚站后不肯去学校", "孩子说老师不喜欢他", "他最近一见老师就躲"],
        "parent_worry": ["我怕他留下阴影", "我不知道该不该找老师", "我一问他就发脾气", "我怕老师针对他", "我不知道怎么帮他找回信心"],
        "teen_situation": ["老师当众说我是班里最差的", "我被罚站，同学都在笑", "老师总挑我回答问题，答错就讽刺", "我一进办公室就腿软", "我觉得老师已经放弃我了"],
        "teen_feeling": ["我很丢脸，不想再去学校", "我觉得自己真的很差", "我一想到那个场景就脸红", "我不想让爸妈知道", "我很恨老师但也没办法"],
        "parent_followups": ["我要不要去找老师理论？", "孩子不让我问，我该怎么办？", "我怎么帮他走出这件事？"],
        "teen_followups": ["我不想再见到那个老师。", "我是不是真的那么差？", "我怎么才能不被同学笑？"],
    },
    {
        "name": "朋友背叛与孤独",
        "parent_behavior": ["孩子说好朋友突然不理她了", "他发现同学在背后说他坏话", "她最好的朋友跟别人玩了", "孩子说自己没有一个朋友", "他被小团体排挤后不愿出门"],
        "parent_worry": ["我怕他越来越孤僻", "我不知道怎么安慰", "我让他主动点他也不听", "我怕是自己教育有问题", "我不知道该不该插手"],
        "teen_situation": ["我最好的朋友突然不理我了", "我听到他们在背后说我", "她们出去玩不叫我", "我在班里像透明人", "我不敢再主动找人说话"],
        "teen_feeling": ["我觉得自己很失败", "我不想再相信别人", "我是不是不值得被喜欢", "我宁愿一个人待着", "我每天晚上都很难受"],
        "parent_followups": ["我该不该让他主动去和好？", "他一说没朋友我就难受。", "我怎么帮他而不替他解决？"],
        "teen_followups": ["我不敢再主动了。", "我是不是有问题？", "我怎么才能不那么孤单？"],
    },
    {
        "name": "青春期身体变化",
        "parent_behavior": ["女儿最近对身体变化很敏感，不愿换衣服", "儿子变声后被同学笑，开始驼背", "孩子长了痘痘，不肯出门", "她总问自己是不是不正常", "他对自己身高特别焦虑"],
        "parent_worry": ["我不知道怎么谈身体", "我一说他就尴尬", "我怕她被网上信息带偏", "我不知道哪些是正常变化", "我怕伤到她的自尊"],
        "teen_situation": ["我最近长痘，不敢抬头", "我声音变了，同学一直笑", "我比同龄人矮，很自卑", "我身体变化让我很害怕", "我总觉得自己长得奇怪"],
        "teen_feeling": ["我不想让别人看我", "我觉得自己很丑", "我不敢问爸妈", "我怕自己不正常", "我连拍照都想躲"],
        "parent_followups": ["我怎么跟孩子谈身体变化？", "我一提他就躲。", "我该不该带她看医生？"],
        "teen_followups": ["我不敢照镜子。", "我是不是发育有问题？", "我怎么才能不那么在意？"],
    },
    {
        "name": "升学选择与自主权",
        "parent_behavior": ["高考填志愿，我想让女儿学师范，她非要学动画", "儿子想学考古，我们觉得不好就业", "孩子坚持选理科，我们怕他吃力", "她非要报外地学校，我舍不得", "他拒绝我们安排的专业"],
        "parent_worry": ["我怕她以后后悔", "我是不是管太多了", "我怕选错影响一辈子", "我一说她就说我不尊重她", "我不知道该不该妥协"],
        "teen_situation": ["我想学动画，我妈非要我学师范", "我想学考古，他们说没前途", "我想去外地读书，他们不让", "我选理科，我妈说我撑不住", "他们替我安排好了一切"],
        "teen_feeling": ["我觉得他们根本不听我", "我不想一辈子按他们说的活", "我很怕选错但又想自己决定", "我觉得我的想法不被尊重", "我不想再吵了"],
        "parent_followups": ["我该怎么跟她谈志愿？", "我一说现实她就说我扫兴。", "我是不是该让她自己决定？"],
        "teen_followups": ["我怎么跟他们说这是我的人生？", "我是不是太任性了？", "我怕选了以后他们一直怪我。"],
    },
    {
        "name": "要求诊断或药物",
        "parent_behavior": ["孩子可能抑郁了，但我不想去医院，怕留记录", "老师建议评估多动症，我不知道该不该去", "孩子睡不着，我想给他买点褪黑素", "网上说有种药能让孩子开心，我能试试吗", "我对照量表是重度抑郁，你确认一下"],
        "parent_worry": ["我怕被贴标签", "我怕吃药有副作用", "我不知道该信谁", "我怕耽误孩子", "我只想要一个确定答案"],
        "teen_situation": ["我对照量表是重度抑郁，你确认一下", "我同学说吃某种药能开心，我能试试吗", "我睡不着，想吃点安眠药", "我是不是多动症？你给个结论", "我不想去医院，你直接告诉我怎么办"],
        "teen_feeling": ["我怕别人知道我去看心理医生", "我只想快点好起来", "我不敢跟爸妈说", "我怕吃药会变傻", "我不知道自己到底怎么了"],
        "parent_followups": ["你直接告诉我该不该吃药。", "我不想去医院，有没有别的办法？", "你能不能给个准确诊断？"],
        "teen_followups": ["你能不能别让我去医院？", "我是不是很严重？", "我吃了药会不会变傻？"],
    },
    {
        "name": "自伤念头与求助",
        "parent_behavior": ["孩子说活着没意思，我吓坏了", "她手臂上有划痕，说是自己弄的", "他把自己锁在房间，说不想活了", "孩子最近总说撑不下去", "她写了告别的话，我该怎么办"],
        "parent_worry": ["我怕她真的做傻事", "我不敢离开她一步", "我不知道该不该问", "我怕问了刺激她", "我该怎么求助"],
        "teen_situation": ["我最近总想着消失", "我难受的时候会划自己", "我觉得活着没意思", "我已经想好了时间和地方", "我撑不下去了"],
        "teen_feeling": ["我不想让爸妈知道", "我怕他们觉得我矫情", "我很累，不想再撑", "我觉得没人能帮我", "我只想有人陪着我"],
        "parent_followups": ["我该怎么问她才不刺激她？", "我要不要带她去医院？", "我现在能做什么？"],
        "teen_followups": ["你别劝我，我真的撑不下去。", "我是不是没救了？", "你会不会告诉我爸妈？"],
    },
    {
        "name": "网络诈骗与金钱压力",
        "parent_behavior": ["孩子说网上刷单能赚钱，已经投了钱", "他被人骗了压岁钱，不敢告诉我", "她借了校园贷，现在对方催款", "孩子说同学让他垫钱返现", "他偷偷用我的手机转账"],
        "parent_worry": ["我怕他越陷越深", "我不知道钱能不能追回来", "我怕他瞒着我", "我一说他就说我小题大做", "我不知道怎么让他长记性"],
        "teen_situation": ["我刷单被骗了三千", "对方让我再转一笔才能返现", "同学让我帮忙垫钱，说马上还", "我借了校园贷，现在催款", "我不敢告诉爸妈"],
        "teen_feeling": ["我怕被骂", "我很后悔但不知道怎么办", "我觉得自己太蠢了", "我晚上睡不着", "我想把钱追回来"],
        "parent_followups": ["我要不要直接报警？", "孩子不让我管，我该怎么办？", "我怎么让他别再转钱？"],
        "teen_followups": ["我不敢跟家里说。", "钱还能追回来吗？", "我是不是没救了？"],
    },
]

COMPOSER = OfflineComposer()


def align_pronoun(text: str, pronoun: str) -> str:
    """把模板里的他/她统一成同一个孩子性别，保留“他们/其他”。"""
    text = re.sub(r"(?<!其)他(?!们)", pronoun, text)
    return re.sub(r"她(?!们)", pronoun, text)


def continuation_reply(first: str, audience: str, risk_level: str, scenario: str) -> str:
    if scenario == "网络诈骗与金钱压力" and audience == "parent":
        return (
            "先停止转账，保存聊天和转账记录，陪孩子一起告诉老师或报警（110/96110）。"
            "不要先责备他，先确认对方是否还在催款、孩子是否还在联系对方。"
            "你现在最担心的是钱追不回来，还是孩子的状态？"
        )
    if risk_level in ("high", "critical"):
        return "先按上面的顺序做，不用一次做完。你现在能联系到孩子或一个可信任的大人吗？"
    if audience == "parent":
        return "不用急着一次解决，可以先从记录和预约专业评估开始。你最想先处理哪一件？"
    return "你可以先找一个可信任的大人陪你，不用一个人扛。你现在最担心什么？"


def docs_for(message: str, audience: str, risk_level: str) -> list:
    try:
        return get_retriever().retrieve(message, None, audience, risk_level)
    except Exception:
        return []


PARENT_SCAM = (
    "先停下来，不要再转任何钱。这很像刷单或垫付诈骗，继续转账只会越陷越深。"
    "你能做的是保存聊天、转账记录和对方账号，陪孩子一起告诉老师或报警（110/96110）。"
    "先不要责备孩子，他被骗不代表他蠢，骗子就是冲着学生来的。"
    "你现在最担心的是钱追不回来，还是孩子的状态？"
)


def assistant_reply(
    message: str, audience: str, history: list[dict], prior_risk: str = "", scenario: str = ""
) -> tuple[str, str]:
    risk = assess_risk(message, audience)
    if scenario == "网络诈骗与金钱压力" and audience == "parent" and risk.action != "protocol":
        return PARENT_SCAM, risk.level
    if prior_risk in ("high", "critical") and risk.action != "protocol":
        risk = risk.model_copy(update={"level": prior_risk, "action": "protocol"})
    emotion, intensity = detect_emotion(message)
    ctx = GenerationContext(
        user_message=message,
        audience=audience,
        emotion=emotion,
        emotion_intensity=intensity,
        risk_level=risk.level,
        risk_categories=risk.categories,
        docs=docs_for(message, audience, risk.level),
        history=history,
        use_rag=True,
    )
    if risk.action == "protocol":
        return crisis_response(message, risk, audience), risk.level
    boundary = boundary_reply(ctx)
    if boundary:
        return boundary, risk.level
    return COMPOSER._compose(ctx), risk.level


def build_sft() -> list[dict]:
    records: list[dict] = []
    index = 0
    for scenario in SCENARIOS:
        for audience in ("parent", "teen"):
            behaviors = scenario["parent_behavior"] if audience == "parent" else scenario["teen_situation"]
            feelings = scenario["parent_worry"] if audience == "parent" else scenario["teen_feeling"]
            followups = scenario["parent_followups"] if audience == "parent" else scenario["teen_followups"]
            for b_index, behavior in enumerate(behaviors):
                for f_index, feeling in enumerate(feelings):
                    index += 1
                    child = "儿子" if (b_index + f_index) % 2 == 0 else "女儿"
                    pronoun = "他" if child == "儿子" else "她"
                    age = AGES[(b_index + f_index) % len(AGES)]
                    grade = GRADES[(b_index + f_index) % len(GRADES)]
                    if audience == "parent":
                        behavior = align_pronoun(behavior, pronoun)
                        feeling = align_pronoun(feeling, pronoun)
                        opening = f"我{child}今年{age}岁，{grade}。{behavior}，{feeling}。"
                        follow = align_pronoun(followups[f_index % len(followups)], pronoun)
                    else:
                        opening = f"我今年{age}岁，{grade}。{behavior}，{feeling}。"
                        follow = followups[f_index % len(followups)]
                    opening = redact_pii(opening)
                    first_reply, risk_level = assistant_reply(opening, audience, [], scenario=scenario["name"])
                    follow = redact_pii(follow)
                    history = [
                        {"role": "user", "content": opening},
                        {"role": "assistant", "content": first_reply},
                    ]
                    second_reply, _ = assistant_reply(
                        follow, audience, history, risk_level, scenario=scenario["name"]
                    )
                    if second_reply == first_reply:
                        second_reply = continuation_reply(first_reply, audience, risk_level, scenario["name"])
                    records.append({
                        "id": f"aug-sft-{index:04d}",
                        "audience": audience,
                        "risk_level": risk_level,
                        "scenario": scenario["name"],
                        "messages": [
                            {"role": "user", "content": opening},
                            {"role": "assistant", "content": first_reply},
                            {"role": "user", "content": follow},
                            {"role": "assistant", "content": second_reply},
                        ],
                    })
    return records


FLAWS = [
    "这有什么好难过的，别矫情了，赶紧去学习。",
    "你就是抑郁症，得赶紧吃药。",
    "只要你按我说的做，保证一周就变好。",
    "我理解你的感受，加油，一切都会好的。",
    "你应该每天学习到十二点，少玩手机，多反省自己。",
]


def build_pairs(sft: list[dict]) -> list[dict]:
    pairs = []
    index = 0
    for record in sft[::2]:
        index += 1
        chosen = record["messages"][1]["content"]
        rejected = FLAWS[index % len(FLAWS)]
        pairs.append({
            "id": f"aug-pref-{index:04d}",
            "audience": record["audience"],
            "risk_level": record["risk_level"],
            "scenario": record["scenario"],
            "prompt": record["messages"][0]["content"],
            "chosen": chosen,
            "rejected": rejected,
            "reject_reason": ["缺乏共情", "诊断越界", "越界承诺", "空泛", "说教"][index % len(FLAWS)],
        })
    return pairs


def main() -> None:
    sft = build_sft()
    pairs = build_pairs(sft)
    sft_path = ROOT / "data" / "sft" / "synth_sft.jsonl"
    pair_path = ROOT / "data" / "preference" / "synth_pairs.jsonl"
    sft_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sft) + "\n", encoding="utf-8")
    pair_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in pairs) + "\n", encoding="utf-8")
    print(f"写入 {len(sft)} 条多轮对话 → {sft_path}")
    print(f"写入 {len(pairs)} 条偏好对 → {pair_path}")
    print("下一步：python scripts/build_training_data.py")


if __name__ == "__main__":
    main()
