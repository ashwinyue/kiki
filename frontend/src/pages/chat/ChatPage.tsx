/**
 * Kiki Agent Framework - 聊天页面
 */

import { ChatSidebar } from './ChatSidebar';
import { ChatMain } from './ChatMain';

export function ChatPage() {
  return (
    <div className="chat-page">
      <ChatSidebar />
      <ChatMain />
    </div>
  );
}
