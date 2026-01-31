/**
 * Kiki Agent Framework - 主布局组件
 */

import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useUIStore } from '@/stores';
import { classNames } from '@/utils/classNames';

export function MainLayout() {
  const { sidebarState, isMobile } = useUIStore();

  return (
    <div
      className={classNames(
        'main-layout',
        `sidebar-${sidebarState}`,
        isMobile && 'is-mobile'
      )}
    >
      <Header />
      <div className="layout-body">
        <Sidebar />
        <main className="main-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
