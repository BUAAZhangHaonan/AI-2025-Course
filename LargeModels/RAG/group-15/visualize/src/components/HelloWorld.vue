<template>
  <McLayout class="container">
    <McHeader :title="'MateChat'" :logoImg="'https://matechat.gitcode.com/logo.svg'">
      <template #operationArea>
        <div class="operations">
          <i class="icon-helping"></i>
        </div>
      </template>
    </McHeader>
    <McLayoutContent
      v-if="startPage"
      style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 12px;
      "
    >
      <McIntroduction
        :logoImg="'https://matechat.gitcode.com/logo2x.svg'"
        :title="'大模型幻觉抑制研究'"
        :subTitle="'CoVe验证链算法 + 开放域问答任务可视化展示'"
        :description="description"
      ></McIntroduction>
      <McPrompt
        :list="introPrompt.list"
        :direction="introPrompt.direction"
        class="intro-prompt"
        @itemClick="onSubmit($event.label)"
      ></McPrompt>
    </McLayoutContent>
    <McLayoutContent class="content-container" v-else>
      <template v-for="(msg, idx) in messages" :key="idx">
        <McBubble
          v-if="msg.from === 'user'"
          :variant="'none'"
          :avatarConfig="{
            imgSrc: 'https://matechat.gitcode.com/png/demo/userAvatar.svg',
          }"
        >
          <div class="suggestion-list">
            <div class="suggestion-item"  :style="{ backgroundColor: msg.color }">
              <i class="icon-inform"></i>
              <span class="suggestion-text">{{ msg.content }}</span>
            </div>
          </div>
        </McBubble>
        <McBubble
          v-else
          :varient="'none'"
          :avatarConfig="{ imgSrc: 'https://matechat.gitcode.com/logo.svg' }"
        >
          <div class="suggestion-list">
            <div class="suggestion-item"  :style="{ backgroundColor: msg.color }">
              <i class="icon-mandatory"></i>
              <span class="suggestion-text">{{ msg.content }}</span>
            </div>
          </div>
        </McBubble>
      </template>
    </McLayoutContent>
    <div class="shortcut" style="display: flex; align-items: center; gap: 8px">
      <McPrompt
        v-if="!startPage"
        :list="simplePrompt"
        :direction="'horizontal'"
        style="flex: 1"
        @itemClick="onSubmit($event.label)"
      ></McPrompt>
      <div v-else class="agent-knowledge">
        <el-select v-model="selectedAgent" placeholder="Select" style="width: 20vw">
          <el-option
            v-for="item in agentList"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <span class="agent-knowledge-dividing-line"></span>
        <div class="knowledge-wrapper">
          <i class="icon-operation-log"></i>
          <span>添加知识</span>
        </div>
      </div>
      <Button
        style="margin-left: auto"
        icon="add"
        shape="circle"
        title="新建对话"
        size="md"
        @click="newConversation"
      />
    </div>
    <McLayoutSender>
      <McInput
        :value="inputValue"
        :maxLength="2000"
        @change="(e) => (inputValue = e)"
        @submit="onSubmit"
      >
        <template #extra>
          <div class="input-foot-wrapper">
            <div class="input-foot-left">
              <span v-for="(item, index) in inputFootIcons" :key="index">
                <i :class="item.icon"></i>
                {{ item.text }}
              </span>
              <span class="input-foot-dividing-line"></span>
              <span class="input-foot-maxlength">{{ inputValue.length }}/2000</span>
            </div>
            <div class="input-foot-right">
              <Button
                icon="op-clearup"
                shape="round"
                :disabled="!inputValue"
                @click="inputValue = ''"
                ><span class="demo-button-content">清空输入</span></Button
              >
            </div>
          </div>
        </template>
      </McInput>
    </McLayoutSender>
  </McLayout>
</template>

<script setup lang="ts">
import { ref } from "vue";
import { Button } from "vue-devui/button";
import "vue-devui/button/style.css";

const description = [
  "验证链（CoVe）源自论文 Chain-of-Verification Reduces Hallucination in Large Language Models。",
  "核心思想是：先产出初稿答案，再由模型自拟验证问题并逐一作答，对照初稿排查不一致与事实偏差，最后基于验证结果修订输出，从而显著降低幻觉。"
];
const introPrompt = {
  direction: "horizontal",
  list: [
    {
      value: "quickSort",
      label: "生成BaseLine回答",
      iconConfig: { name: "icon-info-o", color: "steelblue" },
      desc: "模型给出原始答案，作为对照基线。",
    },
    {
      value: "helpMd",
      label: "提出验证问题",
      iconConfig: { name: "icon-star", color: "seagreen" },
      desc: "围绕初稿生成核验清单，聚焦事实与出处。",
    },
    {
      value: "bindProjectSpace",
      label: "回答问题对照修订",
      iconConfig: { name: "icon-priority", color: "darkorange" },
      desc: "逐题作答后比对初稿，标记冲突与疑点，修订最终回答。",
    },
  ],
};
const simplePrompt = [
  {
      value: "quickSort",
      label: "生成BaseLine回答",
      iconConfig: { name: "icon-info-o", color: "steelblue" },
    },
    {
      value: "helpMd",
      label: "提出验证问题",
      iconConfig: { name: "icon-star", color: "seagreen" },
    },
    {
      value: "bindProjectSpace",
      label: "回答问题对照修订",
      iconConfig: { name: "icon-priority", color: "darkorange" },
    },
];
const startPage = ref(true);
const inputValue = ref("");
const inputFootIcons = [
  { icon: "icon-at", text: "智能体" },
  { icon: "icon-standard", text: "词库" },
  { icon: "icon-add", text: "附件" },
];

const messages = ref<any[]>([]);

const newConversation = () => {
  startPage.value = true;
  messages.value = [];
};

const onSubmit = async (evt) => {
  inputValue.value = "";
  startPage.value = false;
  // 用户发送消息
  messages.value.push({
    from: "user",
    content: evt,
    color: color_basic,
  });
  await displayMessage(reply_basic, alert_basic, 3, "model", color_basic)
  
  for(let i = 0; i < questions_verify.length; i++){

    const question_verify = questions_verify[i]
    const reply_verify = replys_verify[i]
    
    if(i == 0){
      await displayMessage(question_verify, alert_verify, 6, "user", color_verify)
      await displayMessage(reply_verify, alert_basic, 2, "model", color_verify)
    }
    else{
      await displayMessage(question_verify, alert_verify, 1, "user", color_verify)
      await displayMessage(reply_verify, alert_basic, 2, "model", color_verify)
    }
  }
  await displayMessage(reply_final, alert_final, 6, "model", color_final)
};

const agentList = ref([
  { label: "CoVe验证链", value: "CoVe验证链", active: true },
]);
const selectedAgent = ref(agentList.value[0]);

const reply_basic = "Here are several species reported as endemic to Oman (primarily reptiles and amphibians): Arabian toad, Asaccus gallagheri, Asaccus platyrhynchus, Acanthodactylus masirae, Hemidactylus luqueorum, Asaccus arnoldi, Dhofar toad, Hemidactylus masirahensis, Asaccus margaritae."

const alert_basic = "模型生成完毕"

const alert_verify = "验证问题发送"

const alert_final = "最终回答生成"

const color_basic = "steelblue"

const color_verify = "seagreen"

const color_final = "darkorange"

const questions_verify = [
  "Is Arabian toad only found in Oman?",
  "Is Asaccus gallagheri only found in Oman?",
  "Is Asaccus platyrhynchus only found in Oman?",
  "Is Acanthodactylus masirae only found in Oman?",
  "Is Hemidactylus luqueorum only found in Oman?",
  "Is Asaccus arnoldi only found in Oman?",
  "Is Dhofar toad only found in Oman?",
  "Is Hemidactylus masirahensis only found in Oman?",
  "Is Asaccus margaritae only found in Oman?"
]

const replys_verify = [
  "Likely no; ranges across parts of the Arabian Peninsula.",
  "Yes; reported as Oman endemic.",
  "Yes; Oman endemic.",
  "Yes; restricted to Masirah Island (Oman).",
  "Yes; described from Oman, endemic.",
  "Yes; Oman endemic.",
  "Yes; endemic to the Dhofar region (southern Oman).",
  "Yes; Masirah Island endemic.",
  "Yes; Oman endemic."
]

const reply_final = "Here are several animals reported as endemic to Oman: Arabian toad, Asaccus gallagheri, Asaccus platyrhynchus, Acanthodactylus masirae, Hemidactylus luqueorum, Asaccus arnoldi, Dhofar toad"

import { ElMessage, rangeArr } from 'element-plus';

const sleep = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

const displayMessage = async(
  response,
  alert,
  delay,
  user,
  color
) => 
{
  await sleep(delay * 1000)
  messages.value.push({
    from: user,
    content: response,
    color: color,})
  ElMessage({
    message: alert,
    type: 'success',
    duration: 2000,
    showClose: true,
  });
};


</script>

<style scoped lang="scss">
.container {
  width: 1000px;
  margin: 20px auto;
  height: calc(100vh - 82px);
  padding: 20px;
  gap: 8px;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 16px;
}

.content-container {
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow: auto;
}

.input-foot-wrapper {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  height: 100%;
  margin-right: 8px;

  .input-foot-left {
    display: flex;
    align-items: center;
    gap: 8px;

    span {
      font-size: 14px;
      line-height: 18px;
      color: #252b3a;
      cursor: pointer;
    }

    .input-foot-dividing-line {
      width: 1px;
      height: 14px;
      background-color: #d7d8da;
    }

    .input-foot-maxlength {
      font-size: 14px;
      color: #71757f;
    }
  }

  .input-foot-right {
    .demo-button-content {
      font-size: 14px;
    }

    & > *:not(:first-child) {
      margin-left: 8px;
    }
  }
}

.agent-knowledge {
  flex: 1;
  display: flex;
  align-items: center;

  .agent-wrapper {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    border-radius: var(--devui-border-radius-full);
    background-color: var(--devui-area);
    cursor: pointer;

    img {
      width: 16px;
      height: 16px;
      margin-right: 4px;
    }

    span {
      font-size: var(--devui-font-size);
      color: var(--devui-text);
      margin-right: 8px;
    }

    i {
      font-size: var(--devui-font-size);
      color: var(--devui-text);
      transition: transform 0.3s ease-in-out;

      &:last-child {
        margin-left: 4px;
      }
    }

    .is-open {
      transform: rotate(180deg);
    }
  }

  .agent-knowledge-dividing-line {
    width: 1px;
    height: 14px;
    margin: 0 12px;
    background-color: var(--devui-line);
  }

  .knowledge-wrapper {
    font-size: var(--devui-font-size);
    color: var(--devui-text);
    cursor: pointer;

    span {
      margin-left: 4px;
    }
  }
}

.suggestion-list {
  display: flex;
  flex-direction: column;
  gap: 12px;

  .suggestion-item {
    border: 1px solid var(--devui-dividing-line);
    border-radius: 8px;
    padding: 4px 8px;
    cursor: pointer;

    .suggestion-text {
      margin-left: 8px;
    }

    .icon-inform {
      color: white
    }

    .icon-mandatory {
      color: white
    }

    .icon-publish-new {
      color: white
    }
  }
}
</style>
