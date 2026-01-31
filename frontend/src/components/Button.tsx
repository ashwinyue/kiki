/**
 * Kiki Agent Framework - 按钮组件
 *
 * 参考 WeKnora/TDesign 的按钮设计
 */
import React from 'react';
import { classNames } from '@/utils/classNames';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /** 按钮变体 */
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'text';
  /** 按钮尺寸 */
  size?: 'small' | 'medium' | 'large';
  /** 按钮形状 */
  shape?: 'square' | 'round' | 'circle';
  /** 是否为块级按钮 */
  block?: boolean;
  /** 是否加载中 */
  loading?: boolean;
  /** 加载文字 */
  loadingText?: string;
  /** 图标 */
  icon?: React.ReactNode;
  /** 后置图标 */
  suffix?: React.ReactNode;
  /** 危险状态 */
  danger?: boolean;
}

/**
 * 按钮组件
 */
export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'medium',
  shape = 'square',
  block = false,
  loading = false,
  loadingText,
  icon,
  suffix,
  danger = false,
  disabled,
  children,
  className,
  type = 'button',
  ...props
}) => {
  const isDisabled = disabled || loading;

  return (
    <button
      type={type}
      disabled={isDisabled}
      className={classNames(
        'button',
        `button-${variant}`,
        `button-${size}`,
        `button-${shape}`,
        danger && 'button-danger',
        block && 'button-block',
        isDisabled && 'button-disabled',
        className
      )}
      {...props}
    >
      {loading ? (
        <>
          <span className="button-spinner" />
          {loadingText && <span>{loadingText}</span>}
        </>
      ) : (
        <>
          {icon && <span className="button-icon">{icon}</span>}
          {children && <span className="button-content">{children}</span>}
          {suffix && <span className="button-suffix">{suffix}</span>}
        </>
      )}
    </button>
  );
};

/**
 * 图标按钮组件
 */
export interface IconButtonProps extends Omit<ButtonProps, 'children'> {
  /** 图标 */
  icon: React.ReactNode;
  /** 提示文本 */
  tooltip?: string;
}

export const IconButton: React.FC<IconButtonProps> = ({
  icon,
  tooltip,
  size = 'medium',
  variant = 'ghost',
  ...props
}) => {
  const button = (
    <Button
      variant={variant}
      size={size}
      shape="circle"
      aria-label={tooltip}
      {...props}
    >
      {icon}
    </Button>
  );

  if (tooltip) {
    return (
      <div className="icon-button-wrapper">
        {button}
        <span className="icon-button-tooltip">{tooltip}</span>
      </div>
    );
  }

  return button;
};

export default Button;
